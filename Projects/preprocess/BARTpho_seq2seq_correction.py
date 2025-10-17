import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv
from tqdm import tqdm
import re
import unicodedata
import difflib
import signal, sys

# ---------------- LOAD .env ----------------
# Giả sử file .env nằm ở thư mục gốc của project, một cấp trên thư mục chứa script
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    load_dotenv() # Tải từ thư mục hiện tại nếu không tìm thấy ở trên

# ---------------- CONFIG ----------------
MODEL = "vinai/bartpho-syllable"
INPUT_CSV = "./vihallu-public-test.csv"
OUTPUT_CSV = "./fixed-vihallu-public-test-BARTpho-seq2seq.csv" # Đổi tên file output cuối cùng

correction_cache = {}
SAMPLE_DEBUG_N = 20

COLUMNS_TO_FIX = ['context', 'prompt', 'response']
MAX_INPUT_LENGTH = 1024

# thresholds <<< THAY ĐỔI CUỐI CÙNG: Tăng ngưỡng để an toàn hơn
# Điều này sẽ làm giảm số lượng thay đổi được chấp nhận, nhưng tăng độ tin cậy.
ACCEPT_SIMILARITY_THRESHOLD = 0.90 # Tăng từ 0.85
LENGTH_CHANGE_ALLOWED_RATIO = 0.15 # Giảm từ 0.2
STRICT_BASE_SIMILARITY_THRESHOLD = 0.95 # Tăng từ 0.92
LENIENT_BASE_SIMILARITY_THRESHOLD = 0.88 # Giữ nguyên

# ---------------- Device ----------------
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ctrl+C handler
df_global = None
def signal_handler(sig, frame):
    print("\n\n[Ctrl+C] Đang lưu tiến trình...")
    if df_global is not None:
        df_global.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"Đã lưu thành công vào {OUTPUT_CSV}")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

# ---------- tokenizer/model setup ----------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device)
model.eval()
print("Model loaded:", MODEL)

# ---------- Utilities ----------
def remove_diacritics_and_punct(s: str) -> str:
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = unicodedata.normalize('NFC', s)
    s = re.sub(r'[^\w\s]', '', s)
    return s.lower()

def string_similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

# ---------- Prompt ----------
def make_correction_prompt(original_text: str) -> str:
    return original_text

# ---------- Generation ----------
def generate_from_model(prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                       max_length=MAX_INPUT_LENGTH).to(device)
    
    input_len = inputs.input_ids.shape[1]
    max_new_tokens = min(int(input_len * 1.5) + 10, MAX_INPUT_LENGTH)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected.strip()

# ---------- Correction Function ----------
def correct_vietnamese_spelling(text, row_id, col_name, debug_print=False):
    original_text = "" if pd.isna(text) else str(text).strip()
    if not original_text:
        return text

    cache_key = (col_name, original_text)
    if cache_key in correction_cache:
        return correction_cache[cache_key]

    prompt = make_correction_prompt(original_text)
    corrected_text = generate_from_model(prompt)

    # Hậu xử lý để loại bỏ tiền tố thừa do model tạo ra
    prefix_to_remove = "Sửa lỗi chính tả và ngữ pháp:"
    if corrected_text.startswith(prefix_to_remove):
        corrected_text = corrected_text[len(prefix_to_remove):].strip()

    accepted = False
    reason = "not_changed"
    final_text = original_text
    
    if corrected_text and corrected_text.lower() != original_text.lower():
        orig_base = remove_diacritics_and_punct(original_text)
        corr_base = remove_diacritics_and_punct(corrected_text)
        base_sim = string_similarity(orig_base, corr_base)

        # Áp dụng luật siêu nghiêm ngặt cho cột 'context'
        if col_name == 'context':
            # Chỉ chấp nhận nếu nội dung gốc (không dấu) được bảo toàn 100%
            if orig_base != corr_base:
                reason = "base_text_altered_in_context"
            # Vẫn kiểm tra độ dài để tránh các lỗi không mong muốn
            elif abs(len(corrected_text) - len(original_text)) / max(1, len(original_text)) > LENGTH_CHANGE_ALLOWED_RATIO:
                 reason = f"length_changed_too_much (orig: {len(original_text)}, corr: {len(corrected_text)})"
            else:
                accepted = True
                reason = "accepted_context_spelling_fix"
                final_text = corrected_text
        # Giữ luật cũ cho các cột 'prompt' và 'response'
        else:
            current_base_threshold = STRICT_BASE_SIMILARITY_THRESHOLD
            if base_sim < current_base_threshold:
                reason = f"base_similarity_too_low_{base_sim:.2f}"
            elif string_similarity(original_text, corrected_text) < ACCEPT_SIMILARITY_THRESHOLD:
                reason = f"low_similarity_{string_similarity(original_text, corrected_text):.2f}"
            elif abs(len(corrected_text) - len(original_text)) / max(1, len(original_text)) > LENGTH_CHANGE_ALLOWED_RATIO:
                reason = f"length_changed_too_much (orig: {len(original_text)}, corr: {len(corrected_text)})"
            else:
                accepted = True
                reason = "accepted_change"
                final_text = corrected_text

    if debug_print:
        print("\n--- DEBUG SAMPLE ---")
        print(f"ID: {row_id} | COLUMN: {col_name}")
        print(f"ORIGINAL: \n{original_text}")
        print(f"CORRECTED: \n{corrected_text}")
        print(f"RESULT: {'ACCEPTED' if accepted else 'REJECTED'}")
        if not accepted:
            print(f"REASON: {reason}")
        print(f"FINAL TEXT: \n{final_text}")
        print("-" * 20)

    correction_cache[cache_key] = final_text
    return final_text

# ---------------- Main loop ----------------
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Không tìm thấy file '{INPUT_CSV}'.")

df = pd.read_csv(INPUT_CSV)
df_global = df
corrected_ids = set()

for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
    current_id = row['id']
    for col in COLUMNS_TO_FIX:
        original_text = row[col]
        debug_flag = (index < SAMPLE_DEBUG_N)
        corrected_text = correct_vietnamese_spelling(original_text, current_id, col, debug_print=debug_flag)
        if str(original_text) != str(corrected_text):
            corrected_ids.add(current_id)
            df.at[index, col] = corrected_text

print(f"\nĐã hoàn thành. Có {len(corrected_ids)} dòng được thay đổi.")
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\nOutput đã được lưu vào {OUTPUT_CSV}")

