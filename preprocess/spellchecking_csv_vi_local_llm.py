# fix_vietnamese_local_strict_spelling_only_fixed.py
# Fixed: removed hallucination labeling — now only spelling/diacritics/spacing corrections remain.

import os
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from underthesea import text_normalize
import Levenshtein
from dotenv import load_dotenv
from tqdm import tqdm
import re
import unicodedata
from huggingface_hub import snapshot_download

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
OUTPUT_CSV = "./fixed-vihallu-public-test.csv"
COLUMNS_TO_FIX = ['context', 'prompt', 'response']

FEW_SHOT = [
    ("Tôi oline học hôm nay", "Tôi online học hôm nay"),
    ("Đây là 1 ví dụ sai dau", "Đây là một ví dụ sai dấu"),
    ("Toi dang online hoc.", "Tôi đang online học."),
    ("Day la mot vi du sai dau", "Đây là một ví dụ sai dấu"),
    ("Yo ngua cua viec to chuc", "Ý nghĩa của việc tổ chức"),
    # Bổ sung few-shot từ dataset thực tế
    ("Ai quản lí và vân hành việnn bảo tàng ở Checkpoint Charrliee?", 
     "Ai quản lý và vận hành viện bảo tàng ở Checkpoint Charlie?"),
    ("Jacks0n đã thể hiên sự ngưởng mộ của mình vơi Dianaa như thế nao?", 
     "Jackson đã thể hiện sự ngưỡng mộ của mình với Diana như thế nào?"),
    ("Sau trân đấnh nào thì Kiev rơi vào tay Gdiminas?", 
     "Sau trận đánh nào thì Kiev rơi vào tay Gediminas?"),
]

MAX_INPUT_LENGTH = 1024
MAX_NEW_TOKENS_CORRECTION = 512
DETERMINISTIC_CORRECTION = True

# Single thresholds / dicts (provide defaults)
ACCEPT_CORRECTION_THRESHOLD = {
    "context": 0.90,
    "prompt": 0.90,
    "response": 0.90
}
LENGTH_CHANGE_ALLOWED = {
    "context": 0.12,
    "prompt": 0.12,
    "response": 0.12
}
CHAR_RATIO_THRESHOLD = {
    "context": 0.9,
    "prompt": 0.9,
    "response": 0.9
}

EDIT_RATIO_THRESHOLD = 0.80

# debug outputs
correction_cache = {}
FLAGGED_REVIEW_CSV = "flagged_for_manual_review.csv"
DEBUG_LOG_CSV = "debug_corrections.csv"
SAMPLE_DEBUG_N = 20  # print console debug for first N samples (set 0 to disable)

# ---------------- device info ----------------
print("CUDA available:", torch.cuda.is_available())

# print thresholds for debugging
print("\n=== DEBUG: Thresholds ===")
print("MAX_INPUT_LENGTH:", MAX_INPUT_LENGTH)
print("MAX NEW TOKEN CORRECTION:", MAX_NEW_TOKENS_CORRECTION)
print("ACCEPT_CORRECTION_THRESHOLD:", ACCEPT_CORRECTION_THRESHOLD)
print("LENGTH_CHANGE_ALLOWED:", LENGTH_CHANGE_ALLOWED)
print("CHAR_RATIO_THRESHOLD:", CHAR_RATIO_THRESHOLD)
print("EDIT_RATIO_THRESHOLD:", EDIT_RATIO_THRESHOLD)
print("DEBUG_LOG_CSV:", DEBUG_LOG_CSV, " SAMPLE_DEBUG_N:", SAMPLE_DEBUG_N)
print("====================================\n")

# ---------- tokenizer/model kwargs ----------
tokenizer_kwargs = {"use_fast": False, "token": HF_TOKEN, "padding_side": "right", "tokenizer_type": 'llama'}
model_kwargs = {"pretraining_tp": 1, "token": HF_TOKEN}

# ---------- helper load ----------
def load_tokenizer_local_or_remote(model_id):
    try:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True, **{k: v for k, v in tokenizer_kwargs.items() if k != "token"})
        tok.bos_token_id = 1
        return tok, "local_cache"
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        tok.bos_token_id = 1
        return tok, "hf_download"

def load_model_local_or_snapshot(model_id):
    cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
    try:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    except Exception:
        bnb_config = None

    loading_strategies = []
    if bnb_config is not None:
        loading_strategies.append({"kwargs": {"quantization_config": bnb_config, "device_map": "auto", "token": HF_TOKEN, "cache_dir": cache_dir}})
    loading_strategies.append({"kwargs": {"torch_dtype": torch.bfloat16, "device_map": "auto", "token": HF_TOKEN, "cache_dir": cache_dir}})
    loading_strategies.append({"kwargs": {"torch_dtype": torch.float16, "device_map": "auto", "token": HF_TOKEN, "cache_dir": cache_dir}})

    last_e = None
    for strat in loading_strategies:
        try:
            m = AutoModelForCausalLM.from_pretrained(model_id, **strat['kwargs'])
            try:
                model_path = snapshot_download(model_id, token=HF_TOKEN)
            except Exception:
                model_path = cache_dir
            return m, model_path
        except Exception as e:
            last_e = e
            continue
    raise RuntimeError("Cannot load model") from last_e

# ---------- Load tokenizer & model ----------
print("Loading tokenizer (local-first)...")
tokenizer, _ = load_tokenizer_local_or_remote(MODEL)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.bos_token_id = 1

print("Loading model (local-first)...")
model, model_source = load_model_local_or_snapshot(MODEL)
model.eval()
model_is_bnb = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
if model_is_bnb:
    model_device = None
else:
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Model source:", model_source)

# ---------- Utilities: diacritics / token checks ----------
def remove_diacritics(s: str) -> str:
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    return unicodedata.normalize('NFC', s)

def tokens_basic(s: str):
    return re.findall(r"[\w\d]+", s, flags=re.UNICODE)

def token_base_equiv(a: str, b: str) -> bool:
    ta = tokens_basic(a)
    tb = tokens_basic(b)
    if len(ta) != len(tb):
        return False
    for xa, xb in zip(ta, tb):
        if remove_diacritics(xa).lower() != remove_diacritics(xb).lower():
            return False
    return True

def punctuation_seq(s: str) -> str:
    return ''.join([c for c in s if (not c.isalnum() and not c.isspace())])

def digits_equal(a, b):
    return re.sub(r"\D", "", str(a)) == re.sub(r"\D", "", str(b))

def normalize_spaces(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def is_diacritic_or_spacing_only_change(orig: str, corr: str, char_ratio_threshold: float = 0.92):
    if orig is None or corr is None:
        return False, 0.0
    a = remove_diacritics(normalize_spaces(orig)).lower()
    b = remove_diacritics(normalize_spaces(corr)).lower()
    if a == b:
        return True, 1.0
    try:
        ratio = Levenshtein.ratio(a, b)
    except Exception:
        ratio = 0.0
    if ratio >= char_ratio_threshold:
        return True, ratio
    ta = re.findall(r"[\w\d]+", a, flags=re.UNICODE)
    tb = re.findall(r"[\w\d]+", b, flags=re.UNICODE)
    if abs(len(ta)-len(tb)) <= 1 and ratio >= (char_ratio_threshold - 0.05):
        return True, ratio
    return False, ratio

# ---------- Prompt helpers ----------
def make_correction_prompt(original_text, few_shot_examples=None):
    fs = ""
    if few_shot_examples:
        for o, c in few_shot_examples:
            fs += f"Ví dụ:\nVăn bản gốc: {o}\nVăn bản sửa: {c}\n\n"
    prompt = (
        "Bạn là một MÁY SỬA LỖI CHÍNH TẢ tiếng Việt (được dùng cho tiền xử lý dữ liệu)."
        "\nQUY TẮC NGHIÊM NGẶT (PHẢI TUÂN THỦ NGOẠI TRỪ KHI BỊ YÊU CẦU ĐẶC BIỆT):\n"
        "1) CHỈ sửa: lỗi chính tả, dấu thanh (dấu), lỗi đánh máy (spacing/không cách/ thừa khoảng trắng).\n"
        "2) KHÔNG được thêm, bớt, đổi thứ tự hay thay đổi loại từ, tên riêng, số, hay ý nghĩa.\n"
        "3) KHÔNG được thay đổi hay di chuyển dấu câu (.,?;:—...).\n"
        "4) Nếu để sửa sẽ **thay đổi** từ, số từ, hoặc ý nghĩa → TRẢ VỀ **CHÍNH XÁC** VĂN BẢN GỐC (không sửa).\n"
        "5) Trả về **duy nhất** một dòng: văn bản đã sửa. **KHÔNG** thêm chú thích, giải thích, hay dấu ngoặc kép.\n\n"
        f"{fs}"
        f"Văn bản gốc: {original_text}\n"
        "Văn bản sửa:"
    )
    return prompt


# ---------- Generation wrapper ----------
def generate_from_model(prompt, max_new_tokens=MAX_NEW_TOKENS_CORRECTION, do_sample=False):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH)
    input_len = inputs["input_ids"].shape[1]
    try:
        if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = model_device
    except Exception:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception:
        pass

    if do_sample:
        gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=True, temperature=0.0, top_k=1, top_p=0.9, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    else:
        gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)

    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return decoded

# ---------- Main correction function (STRICT checks) ----------
def correct_vietnamese_spelling_local(text, row_id, col_name, debug_print=False):
    original_text = "" if text is None else str(text)
    if original_text.strip() == "":
        return text

    if col_name == "context":
        return text

    cache_key = ("CORRECT_STRICT", col_name, original_text)
    if cache_key in correction_cache:
        return correction_cache[cache_key]

    try:
        norm = text_normalize(original_text)
    except Exception:
        norm = original_text

     # 1. Tạo prompt với "mồi" như đã đề xuất
    prompt = make_correction_prompt(original_text, FEW_SHOT)
    
    try:
        decoded = generate_from_model(prompt, max_new_tokens=MAX_NEW_TOKENS_CORRECTION, do_sample=False)
        
        # 2. BƯỚC QUAN TRỌNG: Cắt bỏ chính xác phần prompt khỏi đầu ra
        # Thao tác này đảm bảo rằng "Văn bản đã sửa:" không bao giờ là một phần của kết quả
        if decoded.startswith(prompt):
            corrected = decoded[len(prompt):].strip()
        else:
            # Nếu có lỗi gì đó, dùng lại phương pháp cũ để dọn dẹp
            corrected = decoded.strip()
            corrected = re.sub(r"(Văn bản (đã được )?sửa:|Văn bản gốc:|Kết quả:|Đáp án:)", "", corrected, flags=re.IGNORECASE)
        
        # 3. Các bước xử lý còn lại giữ nguyên
        # Chỉ lấy dòng cuối cùng, bỏ phần giải thích nếu có
        corrected = corrected.strip().splitlines()[-1].strip()
        corrected = re.sub(r"^(Trả lời:|Answer:)\s*", "", corrected, flags=re.IGNORECASE)
        

        # Nếu model tuôn quá dài (> 2 lần số từ gốc) → fallback
        if len(corrected.split()) > len(original_text.split()) * 2:
            corrected = norm
        
        if not corrected:
            corrected = norm
    except Exception as e:
        correction_cache[cache_key] = original_text
        if debug_print:
            print(f"Fallback due to generation error at ID {row_id}, col {col_name}: {e}, flush=True")
        return original_text

    if corrected == original_text:
        correction_cache[cache_key] = original_text
        return original_text

    debug_reason = None
    debug_char_ratio = None
    accepted = False

    if not digits_equal(original_text, corrected):
        debug_reason = "digits_changed"
        correction_cache[cache_key] = original_text
        accepted = False
    else:
        # check số từ thay đổi nhiều
        orig_tokens = original_text.strip().split()
        corr_tokens = corrected.strip().split()
        if len(orig_tokens) != len(corr_tokens):
            debug_reason = f"word_count_changed_{len(orig_tokens)}->{len(corr_tokens)}"
            correction_cache[cache_key] = original_text
            accepted = False
        else:
            # check dấu câu cuối
            if original_text.strip() and corrected.strip():
                if original_text.strip()[-1] in ".?!…" and corrected.strip()[-1] != original_text.strip()[-1]:
                    debug_reason = "punctuation_end_changed"
                    correction_cache[cache_key] = original_text
                    accepted = False
                else:
                    col_char_thresh = CHAR_RATIO_THRESHOLD.get(col_name, 0.92)
                    ok_diacritic, char_ratio = is_diacritic_or_spacing_only_change(original_text, corrected, char_ratio_threshold=col_char_thresh)
                    debug_char_ratio = char_ratio
                    if not ok_diacritic:
                        debug_reason = "tokens_changed_beyond_diacritics"
                        correction_cache[cache_key] = original_text
                        accepted = False
                    else:
                        if punctuation_seq(original_text) != punctuation_seq(corrected):
                            debug_reason = "punctuation_changed"
                            correction_cache[cache_key] = original_text
                            accepted = False
                        else:
                            try:
                                ratio = Levenshtein.ratio(original_text, corrected)
                            except Exception:
                                ratio = 1.0 if original_text == corrected else 0.0

                            col_sim_threshold = ACCEPT_CORRECTION_THRESHOLD.get(col_name, 0.90)
                            if ratio < col_sim_threshold:
                                debug_reason = f"similarity_below_threshold_{ratio:.2f}<{col_sim_threshold}"
                                correction_cache[cache_key] = original_text
                                accepted = False
                            else:
                                length_change_ratio = abs(len(corrected) - len(original_text)) / max(1, len(original_text))
                                allowed_len = LENGTH_CHANGE_ALLOWED.get(col_name, 0.15)
                                if length_change_ratio > allowed_len:
                                    debug_reason = f"length_change_too_big_{length_change_ratio:.2f}>{allowed_len}"
                                    correction_cache[cache_key] = original_text
                                    accepted = False
                                else:
                                    debug_reason = "accepted"
                                    accepted = True
                                    correction_cache[cache_key] = corrected

    if debug_print:
        print("\n--- DEBUG SAMPLE ---", flush=True)
        print("ID:", row_id, "COLUMN:", col_name, flush=True)
        print("ORIGINAL:", flush=True)
        print(original_text, flush=True)
        print("\nCORRECTED (model returned):", flush=True)
        print(corrected, flush=True)
        print("\nRESULT: ", "ACCEPTED" if accepted else "REJECTED", flush=True)
        print("REASON:", debug_reason, flush=True)
        if debug_char_ratio is not None:
            print("CHAR_RATIO (base-string ratio):", f"{debug_char_ratio:.3f}", flush=True)
        try:
            sim = Levenshtein.ratio(original_text, corrected)
            print("LEVENSHTEIN SIMILARITY:", f"{sim:.3f}", flush=True)
        except Exception:
            pass

    try:
        df_row = pd.DataFrame([{
            "id": row_id,
            "column": col_name,
            "original": original_text,
            "corrected_model": corrected,
            "accepted": accepted,
            "reason": debug_reason,
            "char_ratio": debug_char_ratio,
            "lev_ratio": Levenshtein.ratio(original_text, corrected)
        }])
        if not os.path.exists(DEBUG_LOG_CSV):
            df_row.to_csv(DEBUG_LOG_CSV, index=False, encoding="utf-8-sig")
        else:
            df_row.to_csv(DEBUG_LOG_CSV, index=False, mode="a", header=False, encoding="utf-8-sig")
    except Exception as e:
        if debug_print:
            print("Warning: cannot write debug csv:", e)

    return correction_cache.get(cache_key, original_text)

# ---------------- Main loop ----------------
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Không tìm thấy file '{INPUT_CSV}'.")

df = pd.read_csv(INPUT_CSV)
df_orig = df.copy()

corrected_ids = set()
flagged_changes = []

for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
    current_id = row['id']
    for col in COLUMNS_TO_FIX:
        original_text = row[col]
        if pd.isna(original_text):
            continue
        debug_flag = (index < SAMPLE_DEBUG_N)
        corrected_text = correct_vietnamese_spelling_local(str(original_text), current_id, col, debug_print=debug_flag)
        try:
            ratio = Levenshtein.ratio(str(original_text), str(corrected_text))
        except Exception:
            ratio = 1.0 if str(original_text) == str(corrected_text) else 0.0
        if corrected_text != str(original_text):
            corrected_ids.add(current_id)
            df.at[index, col] = corrected_text
            flagged_changes.append({
                "id": current_id,
                "column": col,
                "original": str(original_text),
                "corrected": str(corrected_text),
                "levenshtein_ratio": ratio
            })

if len(corrected_ids) > 0:
    print(f"\nĐã sửa {len(corrected_ids)} samples")
    if flagged_changes:
        flagged_df = pd.DataFrame(flagged_changes)
        flagged_df.to_csv(FLAGGED_REVIEW_CSV, index=False, encoding="utf-8-sig")
        print(f"Flagged {len(flagged_changes)} corrections saved to {FLAGGED_REVIEW_CSV}")

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\nOutput saved to {OUTPUT_CSV}")
