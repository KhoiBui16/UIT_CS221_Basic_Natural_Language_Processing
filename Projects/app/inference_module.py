# app/inference_module.py
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
from typing import List, Dict, Union
import os
import argparse

# --- CẤU HÌNH NHÃN (BẮT BUỘC GIỐNG FILE TRAIN) ---
ID2LABEL = {
    0: "no",  # Phù hợp, không có ảo giác
    1: "extrinsic",  # Ảo giác ngoại sinh
    2: "intrinsic",  # Ảo giác nội sinh
}


class NLIInferenceDataset(Dataset):
    def __init__(
        self, premises: List[str], hypotheses: List[str], tokenizer, max_len: int = 512
    ):
        self.premises = premises
        self.hypotheses = hypotheses
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, idx):
        return {
            "premise": str(self.premises[idx]),
            "hypothesis": str(self.hypotheses[idx]),
        }


def collate_fn_nli(batch, tokenizer, max_len, device):
    """
    Custom collate function để tokenize ngay trong DataLoader.
    """
    premises = [item["premise"] for item in batch]
    hypotheses = [item["hypothesis"] for item in batch]

    encoding = tokenizer.batch_encode_plus(
        list(zip(premises, hypotheses)),
        padding=True,  # Dynamic padding theo batch
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        return_attention_mask=True,
    )
    return {
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device),
    }


def format_nli_input(df: pd.DataFrame) -> pd.DataFrame:
    """Chuyển đổi dataframe về đúng định dạng NLI lúc train."""
    required_cols = ["context", "prompt", "response"]
    for col in required_cols:
        if col not in df.columns:
            # Hỗ trợ trường hợp file csv dùng tên cột viết hoa hoặc viết tắt
            df.rename(
                columns={col.capitalize(): col for col in df.columns}, inplace=True
            )
            # Nếu vẫn không thấy, báo lỗi
            if col not in df.columns:
                raise ValueError(
                    f"Thiếu cột bắt buộc: '{col}'. Các cột hiện có: {list(df.columns)}"
                )

    df["premise"] = (
        "Câu hỏi: "
        + df["prompt"].astype(str)
        + " Ngữ cảnh: "
        + df["context"].astype(str)
    )
    df["hypothesis"] = df["response"].astype(str)
    return df


# Cache global
_LOADED_MODELS = {}


def load_model_and_tokenizer(model_path_or_repo: str, device: torch.device):
    global _LOADED_MODELS
    if model_path_or_repo in _LOADED_MODELS:
        return _LOADED_MODELS[model_path_or_repo]

    print(f"⏳ Đang load model từ: {model_path_or_repo}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_repo)
        model = AutoModelForSequenceClassification.from_pretrained(model_path_or_repo)
        model.to(device)
        model.eval()
        _LOADED_MODELS[model_path_or_repo] = (model, tokenizer)
        print(f"✅ Load model {model_path_or_repo} thành công!")
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Không thể load model {model_path_or_repo}. Lỗi: {e}")


def run_inference(
    model, tokenizer, df: pd.DataFrame, device: torch.device, batch_size: int = 32
) -> pd.DataFrame:
    """Hàm thực thi inference chính."""
    # Tạo bản sao để không ảnh hưởng df gốc bên ngoài nếu cần giữ nguyên
    df_out = df.copy()

    # Format dữ liệu để lấy premise/hypothesis
    df_formatted = format_nli_input(df_out)

    dataset = NLIInferenceDataset(
        premises=df_formatted["premise"].tolist(),
        hypotheses=df_formatted["hypothesis"].tolist(),
        tokenizer=tokenizer,
    )

    collate_wrapper = lambda batch: collate_fn_nli(batch, tokenizer, 512, device)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_wrapper
    )

    all_probs = []
    all_preds = []

    print(f"🚀 Bắt đầu inference cho {len(df)} mẫu...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferencing"):
            outputs = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    # --- CẬP NHẬT THEO YÊU CẦU: Dùng tên cột 'predict_label' và 'score' ---
    df_out["predict_label"] = [ID2LABEL[pred] for pred in all_preds]
    df_out["score"] = [probs[pred] for probs, pred in zip(all_probs, all_preds)]

    # Xóa các cột trung gian nếu không muốn trả về (tùy chọn)
    if "premise" in df_out.columns:
        del df_out["premise"]
    if "hypothesis" in df_out.columns:
        del df_out["hypothesis"]

    if "label" in df_out.columns:
        del df_out["label"]
    if "label_id" in df_out.columns:
        del df_out["label_id"]

    return df_out


def predict_hallucination(
    df: pd.DataFrame,
    model_name_or_path: str,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> pd.DataFrame:
    device = torch.device(device_str)
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, device)
    return run_inference(model, tokenizer, df, device)


# --- PHẦN CHẠY LOCAL (CLI) ---
if __name__ == "__main__":
    # 1. Xác định thư mục gốc của dự án (giả sử file này nằm trong folder 'app/')
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

    # 2. Thiết lập các đường dẫn mặc định
    DEFAULT_INPUT_CSV = os.path.join(ROOT_DIR, "data", "preprocessed", "test_split.csv")
    DEFAULT_OUTPUT_CSV = os.path.join(ROOT_DIR, "data", "inference_output.csv")
    DEFAULT_MODEL = "KhoiBui/xlm-roberta-large-hallucination-classification"

    parser = argparse.ArgumentParser(
        description="Chạy inference kiểm tra ảo giác từ file CSV."
    )

    # Cập nhật: không bắt buộc --csv nữa, nếu không nhập sẽ dùng DEFAULT_INPUT_CSV
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_INPUT_CSV,
        help=f"Đường dẫn file CSV đầu vào (Mặc định: {DEFAULT_INPUT_CSV})",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HF Repo ID hoặc đường dẫn local (Mặc định: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help=f"File output (Mặc định: {DEFAULT_OUTPUT_CSV})",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Thiết bị chạy (cuda/cpu)",
    )

    args = parser.parse_args()
    print(f"🔧 Cấu hình chạy:")
    print(f"  - Input:  {args.csv}")
    print(f"  - Output: {args.output}")
    print(f"  - Model:  {args.model}")
    print(f"  - Device: {args.device}")

    if not os.path.exists(args.csv):
        print(f"\n❌ Lỗi: Không tìm thấy file đầu vào tại: {args.csv}")
        print("👉 Vui lòng kiểm tra lại đường dẫn hoặc chạy từ thư mục gốc của dự án.")
        exit(1)

    print(f"📂 Đang đọc file: {args.csv}")
    df = pd.read_csv(args.csv)
    try:
        # Đảm bảo thư mục output tồn tại
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

        result_df = predict_hallucination(df, args.model, args.device)
        result_df.to_csv(args.output, index=False, encoding="utf-8-sig")

        print(f"\n✅ Hoàn tất! Kết quả đã lưu tại: {args.output}")
        print("👀 Preview 5 dòng đầu tiên (cột dự đoán):")
        print(result_df[["predict_label", "score"]].head())
    except Exception as e:
        print(f"\n❌ Đã xảy ra lỗi khi chạy inference:\n{e}")

# python app/inference_module.py --csv <đường_dẫn_file_csv> --model <tên_model_hoặc_đường_dẫn> [--output <tên_file_kết_quả>] [--device <cpu/cuda>]
