# dataset_temp_base.py

import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os

class HallucinationDataset(Dataset):
    """Custom Dataset cho bài toán phát hiện ảo giác."""
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def prepare_data(config, logger=None):
    """
    Đọc, tiền xử lý, chia dữ liệu thành tập train/validation và LƯU chúng ra file.
    Trả về: train_df, val_df
    """
    # Fix path logic: nếu config.TRAIN_FILE là path đầy đủ thì dùng trực tiếp, tránh join duplicate
    if os.path.isabs(getattr(config, "TRAIN_FILE", "")) or os.path.exists(getattr(config, "TRAIN_FILE", "")):
        data_path = config.TRAIN_FILE
    else:
        data_path = os.path.join(config.DATA_DIR, config.TRAIN_FILE)
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Đọc thành công {len(df)} mẫu từ {data_path}")
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file dữ liệu tại {data_path}")
        return None, None


    # Chuyển đổi các cột sang string để tránh lỗi khi có giá trị NaN
    df['context'] = df['context'].astype(str)
    df['prompt'] = df['prompt'].astype(str)
    df['response'] = df['response'].astype(str)

    # # Format NLI mới (co the thu cho xnli format nay truoc) => nho dong bo voi file predict
    # premise = "Câu hỏi: " + df['prompt'] + " Ngữ cảnh: " + df['context']
    # hypothesis = df['response']
    
    # # Thêm token ngăn cách một cách rõ ràng
    # df['input_text'] = premise + " </s></s> " + hypothesis
    
    
    # --- FORMAT NLI MỚI - CHỐNG MẤT CHỮ  ---  (co the thu cho xnli format nay truoc) => nho dong bo voi file predict (base line)
    # Cấu trúc: [Prompt] </s></s> [Response] </s></s> [Context]
    # Giúp bảo vệ prompt và response không bị cắt.
    # Hybrid prompt la tot nhat (0.7812)
    df['input_text'] = (
        df['prompt'] + 
        " </s></s> " + 
        df['response'] + 
        " </s></s> " + 
        df['context']
    )
    # --- KET THUC FORMAT NLI MOI ---    
    
    
    # Prompt for QA format
    # df['input_text'] = (
    #     "Câu hỏi: " + df['prompt'].astype(str) + " </s> " +
    #     "Câu trả lời được đưa ra: " + df['response'].astype(str) + " </s></s> " +
    #     "Dựa vào ngữ cảnh sau: " + df['context'].astype(str)
    # )
    
    # Prompt for NLI format
    # premise = df['context'].astype(str)
    # hypothesis = df['response'].astype(str)

    # df['input_text'] = premise + " </s></s> " + hypothesis
    
    
    # In một vài ví dụ để kiểm tra
    print("\n=== KIỂM TRA FORMAT DỮ LIỆU MỚI ===")
    sample = df['input_text'].iloc[0]
    print(f"Mẫu input: {sample}...") # In 300 ký tự đầu để xem cấu trúc

    
    # Ánh xạ nhãn theo logic NLI mới
    df['label_id'] = df['label'].map(config.LABEL_MAP)
    
    # Xử lý các dòng có thể có nhãn null sau khi map
    df.dropna(subset=['label_id'], inplace=True)
    df['label_id'] = df['label_id'].astype(int)

    # Chia train/validation
    train_df, val_df = train_test_split(
        df,
        test_size=config.VALIDATION_SPLIT_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df['label_id']  # Đảm bảo phân bổ nhãn đều
    )

    if logger:
        logger.info(f"Chia dữ liệu: {len(train_df)} mẫu train, {len(val_df)} mẫu validation.")
    
    # --- PHẦN NÂNG CẤP: LƯU FILE RA THƯ MỤC DATA ---
    # Tạo thư mục 'processed' trong 'data' nếu chưa có
    processed_data_dir = os.path.join(config.DATA_DIR, 'processed')
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Định nghĩa đường dẫn file
    train_output_path = os.path.join(processed_data_dir, 'train_split.csv')
    val_output_path = os.path.join(processed_data_dir, 'validation_split.csv')
    
    # Lưu các DataFrame
    train_df.to_csv(train_output_path, index=False, encoding='utf-8-sig')
    val_df.to_csv(val_output_path, index=False, encoding='utf-8-sig')
    

    print(f"✅ Đã lưu tập train vào: {train_output_path}")
    print(f"✅ Đã lưu tập validation vào: {val_output_path}")
    # --- KẾT THÚC PHẦN NÂNG CẤP ---
    
    return train_df, val_df