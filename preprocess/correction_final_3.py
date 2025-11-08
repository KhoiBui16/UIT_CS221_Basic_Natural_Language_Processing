# fix_vietnamese_local_strict_spelling_only_fixed.py
# Final Version: Data-driven few-shot examples and a more comprehensive system prompt.

import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from dotenv import load_dotenv
from tqdm import tqdm
import re
import unicodedata
from huggingface_hub import snapshot_download
import signal
import sys
import difflib
import logging

# ---------------- Suppress Transformers Warning ----------------
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

# ---------------- LOAD .env ----------------
dotenv_path = os.path.join(os.path.dirname(__file__), "..", "envs", ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    load_dotenv()

HF_TOKEN = (
    os.getenv("HUGGING_FACE_TOKEN") or
    os.getenv("HUGGINGFACE_TOKEN") or
    os.getenv("HF_TOKEN") or
    os.getenv("HUNGGING_FACE_TOKEN") or
    None
)

# ---------------- CONFIG ----------------
MODEL = "VietnamAIHub/Vietnamese_llama2_7B_8K_SFT_General_domain"
INPUT_CSV = "./vihallu-public-test.csv"
OUTPUT_CSV = "./fixed-vihallu-public-test_final_3.csv"
COLUMNS_TO_FIX = ['context', 'prompt', 'response']

# *** UPDATE: Few-shot examples được chọn lọc từ chính dữ liệu train/test của bạn ***
FEW_SHOT = [
    # Ví dụ về lỗi gõ tắt và teen code
    ("Nguồn thhu chinh của thuê truc tiếp la nhủg ai?", "Nguồn thu chính của thuế trực tiếp là những ai?"),
    # Ví dụ về lỗi sai dấu và thiếu ký tự
    ("Ý nhĩa cũa viẹc tổ chưc cuôc thi Intervision là gì z?", "Ý nghĩa của việc tổ chức cuộc thi Intervision là gì?"),
    # Ví dụ về viết không dấu
    ("cho biet them thong tin ve su kien nay", "cho biết thêm thông tin về sự kiện này"),
    # Ví dụ về lỗi gõ phím và từ tiếng Anh
    ("Từ Jeew trong tiếng Anh dc bắt nguồn từ tờ nào?", "Từ 'Jew' trong tiếng Anh được bắt nguồn từ từ nào?"),
    # Ví dụ tổng hợp nhiều lỗi
    (" neu ko co su giam sat chat che, hau qua se rat nghiem trong", "nếu không có sự giám sát chặt chẽ, hậu quả sẽ rất nghiêm trọng")
]


MAX_INPUT_LENGTH = 2048

# --- CÁC NGƯỠNG LỌC VẪN GIỮ NGUYÊN TỪ PHIÊN BẢN TRƯỚC ---
WORD_COUNT_TOLERANCE = 1
ACCEPT_SIMILARITY_THRESHOLD = 0.88 
LENGTH_CHANGE_ALLOWED_RATIO = 0.15
STRICT_BASE_SIMILARITY = 0.96
LENIENT_BASE_SIMILARITY = 0.94

# debug outputs
correction_cache = {}
DEBUG_LOG_CSV = "debug_corrections_final_3.csv"
SAMPLE_DEBUG_N = 20

# ---------------- device info ----------------
print("CUDA available:", torch.cuda.is_available())

print("\n=== DEBUG: Thresholds ===")
print(f"STRICT (prompt, response): BASE_SIMILARITY >= {STRICT_BASE_SIMILARITY}")
print(f"LENIENT (context): BASE_SIMILARITY >= {LENIENT_BASE_SIMILARITY}")
print(f"GENERAL: WORD_COUNT_TOLERANCE <= {WORD_COUNT_TOLERANCE}, ACCEPT_SIMILARITY >= {ACCEPT_SIMILARITY_THRESHOLD}")
print("====================================\n")

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
tokenizer_kwargs = {"use_fast": False, "token": HF_TOKEN, "padding_side": "left"}
model_kwargs = {"token": HF_TOKEN}

def load_tokenizer_local_or_remote(model_id):
    try:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True, **{k: v for k, v in tokenizer_kwargs.items() if k != "token"})
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
    
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, "local_cache" if 'local_files_only' in locals() else "hf_download"

def load_model_local_or_snapshot(model_id):
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    loading_strategies = [
        {"kwargs": {"quantization_config": BitsAndBytesConfig(load_in_8bit=True), "device_map": "auto", "token": HF_TOKEN, "cache_dir": cache_dir}},
        {"kwargs": {"torch_dtype": torch.bfloat16, "device_map": "auto", "token": HF_TOKEN, "cache_dir": cache_dir}},
        {"kwargs": {"torch_dtype": torch.float16, "device_map": "auto", "token": HF_TOKEN, "cache_dir": cache_dir}},
    ]
    last_e = None
    for strat in loading_strategies:
        try:
            m = AutoModelForCausalLM.from_pretrained(model_id, **strat['kwargs'])
            return m, snapshot_download(model_id, token=HF_TOKEN, cache_dir=cache_dir) if HF_TOKEN else cache_dir
        except Exception as e:
            last_e = e
    raise RuntimeError("Cannot load model") from last_e

print("Loading tokenizer...")
tokenizer, _ = load_tokenizer_local_or_remote(MODEL)
print("Loading model...")
model, model_source = load_model_local_or_snapshot(MODEL)
model.eval()
try:
    model_device = next(model.parameters()).device
except:
    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Model loaded from:", model_source)

# ---------- Utilities ----------
def remove_diacritics_and_punct(s: str) -> str:
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = unicodedata.normalize('NFC', s)
    s = re.sub(r'[^\w\s]', '', s)
    return s.lower()

def normalize_spaces(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def string_similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

# ---------- Prompt & Generation ----------
def make_correction_prompt_llama2(original_text):
    """
    *** UPDATE: System prompt được làm rõ hơn, định nghĩa các loại lỗi cần sửa ***
    """
    fs_examples = "\n".join([f"Văn bản gốc: \"{o}\"\nVăn bản đã sửa: \"{c}\"" for o, c in FEW_SHOT])
    system_prompt = (
        "Bạn là một chuyên gia hiệu đính tiếng Việt. Nhiệm vụ của bạn là đọc văn bản gốc và sửa lại cho đúng chính tả và ngữ pháp chuẩn mực. "
        "Các loại lỗi cần sửa bao gồm:\n"
        "- Lỗi gõ phím (ví dụ: 'thhu' -> 'thu').\n"
        "- Lỗi sai dấu thanh (ví dụ: 'hoả' -> 'hỏa').\n"
        "- Lỗi viết tắt, teen code (ví dụ: 'ko dc' -> 'không được').\n"
        "- Lỗi viết không dấu (ví dụ: 'toi di hoc' -> 'tôi đi học').\n\n"
        "**QUY TẮC BẮT BUỘC:**\n"
        "1. **GIỮ NGUYÊN Ý NGHĨA:** Tuyệt đối không được thay đổi, thêm, bớt hay tóm tắt nội dung gốc.\n"
        "2. **GIỮ NGUYÊN SỐ LƯỢNG TỪ:** Cố gắng giữ số lượng từ gần bằng nhất có thể so với bản gốc.\n"
        "3. **CHỈ SỬA LỖI:** Nếu văn bản đã đúng, hãy lặp lại y hệt."
    )
    user_prompt = (
        "Dưới đây là các ví dụ:\n"
        f"{fs_examples}\n\n"
        "Bây giờ, hãy sửa văn bản sau đây. Chỉ trả về duy nhất văn bản đã được sửa.\n"
        f"Văn bản gốc: \"{original_text}\""
    )
    return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST] Văn bản đã sửa: \""

def generate_from_model(prompt, original_text_for_token_limit):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH)
    try:
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
    except Exception:
        pass
    
    input_token_length = len(tokenizer.encode(original_text_for_token_limit, add_special_tokens=False))
    
    dynamic_min_new_tokens = int(input_token_length * 0.8)
    dynamic_max_new_tokens = int(input_token_length * 1.5) + 20

    gen_cfg = GenerationConfig(
        min_new_tokens=dynamic_min_new_tokens,
        max_new_tokens=dynamic_max_new_tokens,
        do_sample=False,
        num_beams=3,
        repetition_penalty=1.1,
        early_stopping=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg)
    
    full_decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    
    if "[/INST]" in full_decoded:
        response_part = full_decoded.split("[/INST]")[-1].strip()
        if response_part.lower().startswith("văn bản đã sửa:"):
            response_part = response_part[len("Văn bản đã sửa:"):].strip()
        
        corrected = response_part.split('\n')[0].strip()
        corrected = re.sub(r'^["\']|["\']$', '', corrected)
        if corrected.endswith('"'):
            corrected = corrected[:-1]
        return corrected
    return ""

# ---------- Main correction function ----------
def correct_vietnamese_spelling(text, row_id, col_name, debug_print=False):
    original_text = "" if pd.isna(text) else str(text).strip()
    if not original_text:
        return text

    cache_key = (col_name, original_text)
    if cache_key in correction_cache:
        return correction_cache[cache_key]

    prompt = make_correction_prompt_llama2(original_text)
    corrected_text = generate_from_model(prompt, original_text_for_token_limit=original_text)

    accepted = False
    reason = "not_changed"
    final_text = original_text
    base_sim = 1.0

    if corrected_text and corrected_text != original_text:
        current_base_threshold = STRICT_BASE_SIMILARITY if col_name in ['prompt', 'response'] else LENIENT_BASE_SIMILARITY
        orig_words = normalize_spaces(original_text).split()
        corr_words = normalize_spaces(corrected_text).split()
        orig_base = remove_diacritics_and_punct(original_text)
        corr_base = remove_diacritics_and_punct(corrected_text)
        base_sim = string_similarity(orig_base, corr_base)

        if abs(len(orig_words) - len(corr_words)) > WORD_COUNT_TOLERANCE:
            reason = f"word_count_mismatch_{len(orig_words)}_vs_{len(corr_words)}"
        elif base_sim < current_base_threshold:
            reason = f"base_similarity_too_low_{base_sim:.2f}_(threshold:{current_base_threshold})"
        elif len(original_text) > 0 and abs(len(corrected_text) - len(original_text)) / len(original_text) > LENGTH_CHANGE_ALLOWED_RATIO:
            reason = "length_changed_too_much"
        elif string_similarity(original_text, corrected_text) < ACCEPT_SIMILARITY_THRESHOLD:
            reason = f"low_similarity_{string_similarity(original_text, corrected_text):.2f}"
        elif original_text.endswith('?') and not corrected_text.endswith('?'):
            reason = "question_mark_removed"
        else:
            accepted = True
            reason = "accepted_change"
            final_text = corrected_text

    if debug_print:
        print("\n--- DEBUG SAMPLE ---")
        print(f"ID: {row_id} COLUMN: {col_name}")
        print(f"ORIGINAL: \n{original_text}")
        print(f"\nCORRECTED (model returned): \n{corrected_text}")
        print(f"\nRESULT: {'ACCEPTED' if accepted else 'REJECTED'}")
        print(f"REASON: {reason}")
        if corrected_text and corrected_text != original_text:
            print(f"BASE SIMILARITY: {base_sim:.4f}")
        print(f"FINAL TEXT: \n{final_text}")

    try:
        df_row = pd.DataFrame([{"id": row_id, "column": col_name, "original": original_text, "corrected_model": corrected_text, "final_text": final_text, "accepted": accepted, "reason": reason, "similarity": string_similarity(original_text, corrected_text) if corrected_text else 1.0, "base_similarity": base_sim}])
        log_header = not os.path.exists(DEBUG_LOG_CSV)
        df_row.to_csv(DEBUG_LOG_CSV, index=False, mode='a', header=log_header, encoding="utf-8-sig")
    except Exception as e:
        if debug_print: print(f"Warning: cannot write debug csv: {e}")

    correction_cache[cache_key] = final_text
    return final_text

# ---------------- Main loop ----------------
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Không tìm thấy file '{INPUT_CSV}'.")

df = pd.read_csv(INPUT_CSV)
df_global = df
corrected_ids = set()

if os.path.exists(DEBUG_LOG_CSV):
    os.remove(DEBUG_LOG_CSV)

try:
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
        current_id = row.get('id', index)
        for col in COLUMNS_TO_FIX:
            if col not in row or pd.isna(row[col]):
                continue
            original_text = row[col]
            debug_flag = (index < SAMPLE_DEBUG_N)
            corrected_text = correct_vietnamese_spelling(original_text, current_id, col, debug_print=debug_flag)
            if str(original_text) != str(corrected_text):
                corrected_ids.add(current_id)
                df.at[index, col] = corrected_text
except Exception as e:
    print(f"\nAn error occurred: {e}")
finally:
    print(f"\nĐã xử lý xong. Có {len(corrected_ids)} dòng được thay đổi.")
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nOutput đã được lưu vào {OUTPUT_CSV}")
    print(f"Log chi tiết đã được lưu vào {DEBUG_LOG_CSV}")
    
    print("\n====================================")
    print(f"\nDanh sách những id thay đổi:")
    for idx in corrected_ids:
        print(f"id: {idx}")