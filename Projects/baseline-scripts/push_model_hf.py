import os
import argparse
from turtle import setup
from huggingface_hub import login
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def setup_login(root_dir=""):

    ENVS_PATH = os.path.join(root_dir, "envs", ".env")

    if os.path.exists(ENVS_PATH):
        print(f"Load env from: {ENVS_PATH}")
        load_dotenv(ENVS_PATH)
    else:
        print(f"Env does not exist at: {ENVS_PATH}")
        return

    # get token
    hf_token = os.getenv("HF_TOKEN_WRITE")
    if hf_token:
        print("Token found => Signing in HuggingFace hub...")
        try:
            login(token=hf_token)
            print("Signed in successfully!")
            return True
        except:
            print(f"Not found HF token in {ENVS_PATH}!!!. PLEASE CHECK HF_TOKEN AGAIN.")
            return False


def push_model_to_hf_hub(
    local_model_path, repo_id, commit_message="Upload fine-tuned model"
):
    print(f"Pushing model from {local_model_path} to {repo_id}...")
    if not os.path.exists(local_model_path):
        print(
            f"Local model path: [{local_model_path}] does not exist. Please check the correct path again."
        )
        return

    try:
        print("Loading model and tokenizer...")
        model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)

        print(f"Pushing model to the Hub({repo_id}). This may take a few minutes...")

        model.push_to_hub(repo_id, commit_message=commit_message)
        tokenizer.push_to_hub(repo_id, commit_message=commit_message)

        print(
            f"Push successfully! Your model is online at: https://huggingface.co/{repo_id}"
        )
    except Exception as e:
        print(f"\n❌ Có lỗi xảy ra khi đẩy model: {e}")
        print("👉 Gợi ý kiểm tra:")
        print("   1. Token trong file `envs/.env` có quyền WRITE không?")
        print(
            "   2. Repo ID có đúng không? Nếu repo chưa tồn tại, API sẽ tự tạo nếu token đủ quyền."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Push local HuggingFace fine-tuned model to hub"
    )
    parser.add_argument(
        "--local_path",
        type=str,
        required=True,
        help="Đường dẫn thư mục chứa model local",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="ID của repo trên HF (VD: your-username/my-bert-tuned)",
    )

    parser.add_argument(
        "--msg", type=str, default="Upload fine-tuned model", help="Commit message"
    )

    args = parser.parse_args()
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    setup_login(root_dir=ROOT_DIR)
    push_model_to_hf_hub(args.local_path, args.repo_id, args.msg)


if __name__ == "__main__":
    main()


# --- HƯỚNG DẪN SỬ DỤNG MỚI ---
# 1. Đảm bảo file envs/.env có chứa: HUGGING_FACE_TOKEN=hf_xxxxxxxxx (Token loại WRITE)
# 2. Chạy lệnh (không cần huggingface-cli login nữa):
# python push_model_hf.py --local_path "./models/CafeBERT-tuned" --repo_id "khoibui16/CafeBERT-hallucination-detection"
