# predict_masked_llm_temp.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm.auto import tqdm
import os
import zipfile
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import các cấu hình và module cần thiết
import config_temp_base as config  # Sử dụng config_temp.py
from model_temp_base import get_model_and_tokenizer


class InferenceDataset(Dataset):
    """Dataset cho quá trình inference, không cần nhãn."""

    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }


def predict(model, data_loader, device):
    """Chạy inference trên model và trả về list các dự đoán."""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())

    return all_preds


def main():
    """Hàm chính để chạy pipeline dự đoán và tạo file submission."""
    # 1. Tải dữ liệu test
    test_path = os.path.join(config.DATA_DIR, config.TEST_FILE)
    try:
        test_df = pd.read_csv(test_path)
        print(f"✅ Đọc thành công {len(test_df)} mẫu từ file test: {test_path}")
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file test tại {test_path}")
        return

    # 2. Load model và tokenizer đã huấn luyện
    print(f"Đang tải model đã huấn luyện từ: {config.MODEL_OUTPUT_DIR}")

    # Thay vì gọi get_model_and_tokenizer, hãy load trực tiếp từ thư mục đã lưu
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_OUTPUT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_OUTPUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"🚀 Model đã được load và chuyển sang thiết bị: {device}")

    # 3. Chuẩn bị dữ liệu test

    # # Cập nhật cách tạo input_text  (co the thu cho xnli format nay truoc) => nho dong bo voi file predict

    # premise = "Câu hỏi: " + test_df['prompt'].astype(str) + " Ngữ cảnh: " + test_df['context'].astype(str)
    # hypothesis = test_df['response'].astype(str)
    # test_df['input_text'] = premise + " </s></s> " + hypothesis

    # test nay2 (0.7812)
    # Đồng bộ format input giống hệt file training  (co the thu cho xnli format nay sau) => nho dong bo voi file predict
    test_df["input_text"] = (
        test_df["prompt"]
        + " </s></s> "
        + test_df["response"]
        + " </s></s> "
        + test_df["context"]
    )

    # # Thu prompt moi:
    # test_df['input_text'] = (
    #     "Câu hỏi: " + test_df['prompt'].astype(str) + " </s></s> " +
    #     "Câu trả lời: " + test_df['response'].astype(str) + " </s></s> " +
    #     "Ngữ cảnh: " + test_df['context'].astype(str)
    # )

    # test_df['input_text'] = "Giả thuyết: Khi trả lời câu hỏi test_df['prompt'], câu trả lời test_df['response'] đã phản ánh đúng và không mâu thuẫn với thông tin trong ngữ cảnh. [SEP] Ngữ cảnh: test_df['context']" # Chua thu nghiem

    test_dataset = InferenceDataset(
        texts=test_df["input_text"].to_list(),
        tokenizer=tokenizer,
        max_len=config.MAX_LENGTH,
    )

    # Tăng BATCH_SIZE khi predict để nhanh hơn
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE * 2)

    # 4. Chạy dự đoán
    predictions = predict(model, test_loader, device)

    # 5. Chuyển đổi ID dự đoán thành nhãn dạng chuỗi
    predicted_labels = [config.ID2LABEL[pred_id] for pred_id in predictions]

    # 6. Tạo file submission.csv
    submission_df = pd.DataFrame(
        {"id": test_df["id"], "predict_label": predicted_labels}
    )

    # Tạo thư mục submission nếu chưa có
    if not os.path.exists(config.SUBMISSION_DIR):
        os.makedirs(config.SUBMISSION_DIR)

    csv_path = os.path.join(config.SUBMISSION_DIR, config.SUBMISSION_CSV)
    submission_df.to_csv(csv_path, index=False)
    print(f"✅ Đã tạo thành công file submission: {csv_path}")
    print(submission_df.head(20))

    # 7. Nén thành file submit.zip
    zip_path = os.path.join(config.SUBMISSION_DIR, config.SUBMISSION_ZIP)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_path, arcname=config.SUBMISSION_CSV)
    print(f"✅ Đã nén thành công file zip: {zip_path}")

    print("\n🏁 Quá trình dự đoán và tạo file submission hoàn tất.")


if __name__ == "__main__":
    main()
