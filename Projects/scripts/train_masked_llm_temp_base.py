# train_masked_llm_temp.py

import os
import torch
import numpy as np
import pandas as pd

from dotenv import load_dotenv      
from tqdm.auto import tqdm
from torch.optim import AdamW 
from logger import setup_logger 
from huggingface_hub import login  
from transformers import get_scheduler 
from torch.utils.data import DataLoader
from tqdm.contrib.logging import logging_redirect_tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, classification_report
from functools import partial

import config_temp_base as config
from model_temp_base import get_model_and_tokenizer
from dataset_temp_base import HallucinationDataset, prepare_data

# *** SỬA PHẦN NÀY: Tạo loss function tùy chỉnh kết hợp class weights + label smoothing ***
def loss_fn_with_weights_and_smoothing(logits, labels, class_weights_tensor, label_smoothings):
    """
    Loss function tùy chỉnh kết hợp:
    - Class weights (dynamic computed)
    - Label smoothing
    """
    return torch.nn.functional.cross_entropy(
        logits, 
        labels, 
        weight=class_weights_tensor,
        label_smoothing=label_smoothings
    )

def train_one_epoch(model, train_loader, optimizer, scheduler, device, loss_fn=None, epoch=None, total_epochs=None, logger=None):
    """Huấn luyện mô hình trong một epoch. (dùng loss_fn có class weights)"""
    model.train()
    total_loss = 0
    desc = f"Train" if epoch is None else f"Epoch {epoch}/{total_epochs}"
    
    progress_bar = tqdm(
        train_loader,
        desc=desc,
        # leave=False,
        dynamic_ncols=True,
        mininterval=0.5
    )

    with logging_redirect_tqdm():   # make logger calls safe
        for i, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Zero gradients only at the start of accummulation
            if i % config.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.zero_grad()

            if loss_fn:
                # 1. Dùng loss_fn tùy chỉnh (ví dụ: có class weights)
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
            else:
                # 2. Dùng loss mặc định của model Hugging Face
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            
            # Scale loss for gradient accumulation
            loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            # Track total loss (scale back for logging)
            total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS

            # Update only after accumulating enough steps
            if (i + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0 or (i + 1 == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            progress_bar.set_postfix({'loss': f"{loss.item() * config.GRADIENT_ACCUMULATION_STEPS:.4f}"})
            
            # if logger and (i + 1) % 100 == 0:
            #     current_lr = optimizer.param_groups[0]['lr']
            #     logger.info(f"Step {i+1}, Current LR: {current_lr:.2e}")
            
    if logger: 
        logger.info(f"Training stats: {str(progress_bar)}")

    return total_loss / len(train_loader)

def evaluate(model, val_loader, device, val_df, epoch, loss_fn=None, logger=None):
    model.eval()
    all_preds = []
    all_labels = []
    total_val_loss = 0
    
    # Danh sách để lưu các trường hợp dự đoán sai
    error_records = []
    
    progress_bar = tqdm(val_loader, desc="Evaluating", dynamic_ncols=True)
    
    with torch.no_grad(), logging_redirect_tqdm():
        for i, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if loss_fn:
                # 1. Dùng loss_fn tùy chỉnh
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
            else:
                # 2. Dùng loss mặc định của model
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            
            total_val_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            
            # So sánh dự đoán và nhãn thật để tìm lỗi
            for j in range(len(labels)):
                true_label = labels[j].item()
                pred_label = preds[j].item()
                
                if true_label != pred_label:
                    original_index = i * val_loader.batch_size + j
                    if original_index < len(val_df):
                        original_row = val_df.iloc[original_index]
                        error_records.append({
                            'context': original_row['context'],
                            'prompt': original_row['prompt'],
                            'response': original_row['response'],
                            # SỬA LỖI: Ép kiểu tường minh sang int để Pylance không cảnh báo
                            'true_label': config.ID2LABEL.get(int(true_label), "N/A"),
                            'predicted_label': config.ID2LABEL.get(int(pred_label), "N/A")
                        })
                        
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Sau vòng lặp, nếu có lỗi, lưu chúng vào file CSV
    if logger:
        logger.info(f"Evaluation stats: {str(progress_bar)}")
        logger.info("=== PHÂN TÍCH LỖI (TỔNG HỢP) ===")
        unique_labels = np.unique(all_labels)
        for true_label in unique_labels:
            mask = (np.array(all_labels) == true_label)
            # Thêm kiểm tra để tránh chia cho 0 nếu một lớp không có trong batch
            if np.sum(mask) == 0:
                continue
            pred_for_label = np.array(all_preds)[mask]
            true_label_name = config.ID2LABEL.get(int(true_label), f"Unknown({true_label})")
            logger.info(f"TRUE CLASS: {true_label_name}")
            
            unique_preds_for_label = np.unique(pred_for_label)
            for pred_label in unique_preds_for_label:
                if pred_label != true_label:
                    count = np.sum(pred_for_label == pred_label)
                    percent = count / np.sum(mask) * 100
                    pred_label_name = config.ID2LABEL.get(int(pred_label), f"Unknown({pred_label})")
                    logger.info(f"- Nhầm thành {pred_label_name}: {count} cases ({percent:.1f}%)")
        logger.info("=== KẾT THÚC PHÂN TÍCH LỖI ===")

        error_df = pd.DataFrame(error_records)
        log_dir = os.path.dirname(logger.handlers[0].baseFilename)
        error_log_dir = os.path.join(log_dir, "errors")
        os.makedirs(error_log_dir, exist_ok=True)
        
        error_filename = f"epoch_{epoch}_errors.csv"
        error_filepath = os.path.join(error_log_dir, error_filename)
        
        error_df.to_csv(error_filepath, index=False, encoding='utf-8-sig')
        logger.info(f"Đã lưu {len(error_records)} trường hợp lỗi vào file: {error_filepath}")
    
    avg_val_loss = total_val_loss / len(val_loader) 
    return all_labels, all_preds, avg_val_loss 


def format_params(num_params: int) -> str:
    """Hàm định dạng số tham số thành K, M, B."""
    if num_params >= 1_000_000_000:
        return f"{round(num_params / 1_000_000_000, 2)}B"
    elif num_params >= 1_000_000:
        return f"{round(num_params / 1_000_000, 2)}M"
    elif num_params >= 1_000:
        return f"{round(num_params / 1_000, 2)}K"
    else:
        return str(num_params)



def main():
    dotenv_path = os.path.join(config.ROOT_DIR, "envs", ".env")
    load_dotenv(dotenv_path)
    print(f"dotenv_path: {dotenv_path}")
    
    # lấy HF token để login
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    
    if hf_token:
        print("INFO: Tìm thấy HUGGING_FACE_TOKEN. Đang đăng nhập...")
        login(token=hf_token)
        print("INFO: Đăng nhập Hugging Face thành công.")
    else:
        print("WARNING: Không tìm thấy HUGGING_FACE_TOKEN trong file .env. Một số model có thể yêu cầu đăng nhập.")
    
    """Hàm chính để chạy toàn bộ pipeline huấn luyện."""
    model_name_for_log = os.path.basename(config.MODEL_OUTPUT_DIR)
    logger = setup_logger(model_name_for_log)
    # Sau phần khởi tạo logger, thêm:
    logger.info("=== LABEL MAPPING (NLI) ===")
    logger.info("intrinsic (mâu thuẫn) -> 0 (contradiction)")
    logger.info("extrinsic (trung lập) -> 1 (neutral)")
    logger.info("no (suy ra được) -> 2 (entailment)")
    
    logger.info("=== CONFIGURATION: ===")
    if not all(isinstance(key, str) for key in config.config_vars):
        raise TypeError("All config_vars must be strings")
        
    for key in config.config_vars:
        if hasattr(config, key):
            value = getattr(config, key)
            logger.info(f"{key}: {value}")
        else:
            logger.warning(f"Config variable {key} not found.")

    logger.info("=== Bắt đầu pipeline huấn luyện. === ")
    # 1. Chuẩn bị dữ liệu
    logger.info("Bước 1: Chuẩn bị dữ liệu...")
    train_df, val_df = prepare_data(config, logger=logger)
    if train_df is None or val_df is None:
        logger.error("Dữ liệu không thể được chuẩn bị (train_df hoặc val_df là None). Dừng chương trình.")
        return

    # 2. Tải model(params) và tokenizer
    model, tokenizer = get_model_and_tokenizer(config)
    total_params = sum(p.numel() for p in model.parameters())   # 1. Chỉ tính tổng số tham số, không chia
    formatted_params = format_params(total_params)              # 2. Gọi hàm để có chuỗi đã định dạng đẹp
    logger.info(f"Bước 2: Tải model '{config.MODEL_NAME}'({formatted_params}) và tokenizer...")
    
    # 3. Tạo Dataset và DataLoader
    logger.info("Bước 3: Tạo Dataset và DataLoader...")    
    train_dataset = HallucinationDataset(
        texts=train_df['input_text'].to_list(),
        labels=train_df['label_id'].to_list(),
        tokenizer=tokenizer,
        max_len=config.MAX_LENGTH
    )
    val_dataset = HallucinationDataset(
        texts=val_df['input_text'].to_list(),   
        labels=val_df['label_id'].to_list(),    
        tokenizer=tokenizer,
        max_len=config.MAX_LENGTH
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE,
        num_workers=2
    )
    
    # 4. Thiết lập Huấn luyện
    logger.info("Bước 4: Thiết lập môi trường huấn luyện...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Sử dụng thiết bị: {device}")
    model.to(device)

    # --- BƯỚC KIỂM TRA DỮ LIỆU ---
    if len(train_loader) > 0:
        print("\n=== KIỂM TRA DỮ LIỆU MẪU ===")
        try:
            sample_batch = next(iter(train_loader))
            print("Kích thước input_ids:", sample_batch['input_ids'].shape)
            print("Kích thước attention_mask:", sample_batch['attention_mask'].shape)
            print("Nhãn trong batch:", sample_batch['labels'])
            
            decoded_text = tokenizer.decode(sample_batch['input_ids'][0], skip_special_tokens=False)
            print("\nMột mẫu đã được token hóa và giải mã lại:")
            print("RAW:", decoded_text)
            
            print("\nKiểm tra tokens đặc biệt:")
            print("Special tokens:", tokenizer.special_tokens_map)
            print("Separator token:", tokenizer.sep_token)
            
            # --- SỬA LOGIC PHÂN TÍCH Ở ĐÂY ---
            print("\nPhân tích cấu trúc:")
            parts = decoded_text.split("</s></s>")
            if len(parts) >= 3:
                # Chuẩn hóa danh sách token đặc biệt theo từng kiến trúc
                bos_like = [tokenizer.bos_token, tokenizer.cls_token]
                eos_like = [tokenizer.eos_token, tokenizer.sep_token]
                pad_like = [tokenizer.pad_token]

                # Helper an toàn để xóa các token đặc biệt (bỏ qua None)
                def remove_tokens(s: str, toks):
                    for t in toks:
                        if t:
                            s = s.replace(t, "")
                    return s.strip()

                # 1) Prompt
                prompt_part = remove_tokens(parts[0], bos_like)

                # 2) Response
                response_part = parts[1].strip()

                # 3) Context (ghép lại các phần còn lại)
                context_part = "</s></s>".join(parts[2:]).strip()
                # Loại bỏ eos/sep/pad nếu xuất hiện trong bản decode
                context_part = remove_tokens(context_part, eos_like + pad_like)

                print("PROMPT:", prompt_part)
                print("RESPONSE:", response_part)
                print("CONTEXT:", context_part)
            else:
                print("WARNING: Cấu trúc input không như mong đợi (cần ít nhất 3 phần).")
                print("Số phần tìm thấy:", len(parts))
            print("=" * 50)
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra dữ liệu mẫu: {e}")
    else:
        logger.warning("Train loader rỗng, bỏ qua bước kiểm tra dữ liệu mẫu.")
    # --- KẾT THÚC BƯỚC KIỂM TRA ---
    
    # Loss function setup with dynamic or static weights + Label Smoothing
    logger.info("=== SETTING UP LOSS FUNCTION... ===")
    if config.CLASS_WEIGHTS is None:
        logger.info("Computing dynamic class weights...")
        labels = train_df['label_id'].values
        classes = np.arange(len(config.LABEL_MAP))
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=labels
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        logger.info(f"Computed weights: {class_weights.tolist()}")
    else:
        class_weights_tensor = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float).to(device)
        logger.info(f"Using predefined weights: {config.CLASS_WEIGHTS}")

    # Tích hợp label smoothing vào loss function
    label_smoothing = getattr(config, 'LABEL_SMOOTHING', 0.0)
    logger.info(f"Using label smoothing: {label_smoothing}")

    # SỬA ĐOẠN NÀY: Dùng partial để truyền class_weights_tensor và label_smoothing vào loss_fn
    loss_fn = partial(
        loss_fn_with_weights_and_smoothing,
        class_weights_tensor=class_weights_tensor,
        label_smoothings=label_smoothing
    )
    
    # Optimizer and scheduler setup
    optimizer = AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        eps=config.EPSILON 
    )
    
    # Adjust training steps for gradient accumulation
    num_training_steps = (config.EPOCHS * len(train_loader)) // config.GRADIENT_ACCUMULATION_STEPS
    num_warmup_steps = int(config.TOTAL_STEP_SCALE * num_training_steps)
    logger.info(f"Total training steps: {num_training_steps}, Warm-up steps: {num_warmup_steps}")
    logger.info("=" * 50)

    
    scheduler = get_scheduler(
        "cosine", 
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps,
    )

    # 5. Vòng lặp Huấn luyện
    best_macro_f1 = 0.0
    patience_counter = 0 # bien dem => early stopped khi f1 ko tang them => overfitting

    for epoch in range(config.EPOCHS):
        logger.info(f"--- Epoch {epoch + 1}/{config.EPOCHS} ---")
        
        # Training
        avg_train_loss = train_one_epoch(
            model, 
            train_loader, 
            optimizer, 
            scheduler, 
            device, 
            loss_fn=loss_fn, 
            epoch=epoch + 1, 
            total_epochs=config.EPOCHS, 
            logger=logger
        )
        
        # Validation
        # logger.info("Bắt đầu đánh giá trên tập validation...")
        val_labels, val_preds, avg_val_loss = evaluate(
            model=model, 
            val_loader=val_loader,
            device=device,
            val_df=val_df, 
            epoch=epoch + 1,
            loss_fn=loss_fn, 
            logger=logger
        )
        
        # Metrics
        accuracy = accuracy_score(val_labels, val_preds)
        macro_f1 = f1_score(val_labels, val_preds, average='macro')
        current_lr = optimizer.param_groups[0]['lr']
        
        
        logger.info(f"Current LR: {current_lr:.2e}") 
        logger.info(f"Train loss: {avg_train_loss:.4f}")
        logger.info(f"Vall Loss: {avg_val_loss:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Macro-F1: {macro_f1:.4f}")

        # Detailed classification report
        report = classification_report(
            val_labels, 
            val_preds, 
            target_names=[config.ID2LABEL[i] for i in range(len(config.LABEL_MAP))], 
            digits=4
        )
        logger.info(f"Classification Report:\n{report}")
        
        # Model saving and early stopping
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            patience_counter = 0 
            
            logger.info(f"🎉 Macro-F1 cải thiện. Đang lưu model tốt nhất vào '{config.MODEL_OUTPUT_DIR}'...")
            if not os.path.exists(config.MODEL_OUTPUT_DIR):
                os.makedirs(config.MODEL_OUTPUT_DIR)
                
            # đảm bảo mapping được cùng lưu vào config
            model.config.id2label = config.ID2LABEL
            model.config.label2id = config.LABEL_MAP
            model.save_pretrained(config.MODEL_OUTPUT_DIR)
            tokenizer.save_pretrained(config.MODEL_OUTPUT_DIR)
            logger.info("Lưu model thành công.")
        else:
            patience_counter += 1
            logger.warning(f"Macro-F1 không cải thiện. Patience: {patience_counter}/{config.PATIENCE_LIMIT}")
            if patience_counter >= config.PATIENCE_LIMIT:
                logger.info("Early stopping! Dừng huấn luyện.")
                break

    logger.info("🏁 Quá trình huấn luyện hoàn tất.")
    logger.info(f"Model tốt nhất với Macro-F1 = {best_macro_f1:.4f} đã được lưu tại '{config.MODEL_OUTPUT_DIR}'")

if __name__ == "__main__":
    main()