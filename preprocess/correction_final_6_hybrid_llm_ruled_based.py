# fix_vietnamese_local_strict_spelling_only_fixed.py
# Final Version (v7): Hybrid Approach - Kết hợp Rule-Based tốc độ cao và LLM xử lý sâu.

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
OUTPUT_CSV = "./fixed-vihallu-public-test_final_6.csv"
COLUMNS_TO_FIX = ['context', 'prompt', 'response']

# *** BƯỚC 1: XÂY DỰNG TỪ ĐIỂN SỬA LỖI RULE-BASED ***
# Thêm dấu cách để tránh thay thế trong từ (ví dụ: "trog" trong "strong")
# Từ điển này được xây dựng dựa trên phân tích file train và test của bạn.
RULE_BASED_CORRECTIONS = {
    # Teencode và gõ tắt
    " ko ": " không ", " k ": " không ", " kh ": " không ", " hok ": " không ",
    " dc ": " được ", " đc ": " được ",
    " j ": " gì ", " z?": " vậy?", " z ": " vậy ",
    " nc ": " nước ", " bt ": " biết ", " r ": " rồi ",
    # Lỗi gõ phím và sai dấu phổ biến
    " thhu ": " thu ", " chinh ": " chính ", " thuê ": " thuế ", " truc ": " trực ", " la ": " là ", " nhủg ": " những ",
    " cũa ": " của ", " viẹc ": " việc ", " chưc ": " chức ", " cuôc ": " cuộc ",
    " vân hành ": " vận hành ", " trog ": " trong ",
    " ngta ": " người ta ", " iu ": " yêu ",
    # Các trường hợp cụ thể khác
    " Jeew ": " Jew ",
}

# Few-shot cho LLM để xử lý các ca khó
FEW_SHOT = [
    ("Nguồn thhu chinh của thuê truc tiếp la nhủg ai?", "Nguồn thu chính của thuế trực tiếp là những ai?"),
    ("Ý nhĩa cũa viẹc tổ chưc cuôc thi Intervision là gì z?", "Ý nghĩa của việc tổ chức cuộc thi Intervision là gì?"),
    ("cho biet them thong tin ve su kien nay", "cho biết thêm thông tin về sự kiện này"),
]

MAX_INPUT_LENGTH = 2048

# Giữ nguyên các ngưỡng lọc khắt khe của Ver 5
WORD_COUNT_TOLERANCE = 1
ACCEPT_SIMILARITY_THRESHOLD = 0.92
STRICT_BASE_SIMILARITY = 0.95
LENIENT_BASE_SIMILARITY = 0.90
LENGTH_CHANGE_ALLOWED_RATIO = 0.15 


# debug outputs
correction_cache = {}
DEBUG_LOG_CSV = "debug_corrections_final_6.csv"
SAMPLE_DEBUG_N = 20

# ---------------- device info ----------------
print("CUDA available:", torch.cuda.is_available())
print("\n=== DEBUG: Thresholds (MAXIMUM SAFETY MODE) ===")
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
def load_tokenizer_local_or_remote(model_id):
    try:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True, **{k: v for k, v in tokenizer_kwargs.items() if k != "token"})
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return tok
def load_model_local_or_snapshot(model_id):
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    loading_strategies = [
        {"kwargs": {"quantization_config": BitsAndBytesConfig(load_in_8bit=True), "device_map": "auto", "token": HF_TOKEN, "cache_dir": cache_dir}},
        {"kwargs": {"torch_dtype": torch.bfloat16, "device_map": "auto", "token": HF_TOKEN, "cache_dir": cache_dir}},
    ]
    last_e = None
    for strat in loading_strategies:
        try:
            m = AutoModelForCausalLM.from_pretrained(model_id, **strat['kwargs'])
            return m, snapshot_download(model_id, token=HF_TOKEN, cache_dir=cache_dir) if HF_TOKEN else cache_dir
        except Exception as e: last_e = e
    raise RuntimeError("Cannot load model") from last_e
print("Loading tokenizer...")
tokenizer = load_tokenizer_local_or_remote(MODEL)
print("Loading model...")
model, model_source = load_model_local_or_snapshot(MODEL)
model.eval()
try: model_device = next(model.parameters()).device
except: model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Model loaded from:", model_source)

# ---------- Utilities ----------
def remove_diacritics_and_punct(s: str):
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = unicodedata.normalize('NFC', s)
    s = re.sub(r'[^\w\s]', '', s)
    return s.lower()
def normalize_spaces(s: str): return re.sub(r'\s+', ' ', s).strip()
def string_similarity(a, b): return difflib.SequenceMatcher(None, a, b).ratio()

# ---------- Rule-Based Pre-processor ----------
def rule_based_correction(text):
    """Hàm này sửa các lỗi đơn giản, phổ biến dựa trên từ điển."""
    # Thêm dấu cách ở đầu và cuối để đảm bảo khớp toàn bộ từ và không làm hỏng từ ghép
    padded_text = " " + text + " "
    for wrong, correct in RULE_BASED_CORRECTIONS.items():
        padded_text = padded_text.replace(wrong, correct)
    
    # Bỏ dấu cách đã thêm ở hai đầu
    return padded_text.strip()

# ---------- Prompt & Generation (Giữ nguyên từ Ver 5) ----------
def make_correction_prompt_llama2(original_text):
    fs_examples = "\n".join([f"Văn bản gốc: \"{o}\"\nVăn bản đã sửa: \"{c}\"" for o, c in FEW_SHOT])
    system_prompt = (
        "Bạn là một robot hiệu đính ngôn ngữ siêu chính xác. Nhiệm vụ duy nhất của bạn là sửa lỗi chính tả và lỗi gõ phím.\n"
        "**QUY TẮC QUAN TRỌNG NHẤT:** Văn bản đã sửa phải có SỐ LƯỢNG TỪ CHÍNH XÁC BẰNG với văn bản gốc. Đây là yêu cầu bắt buộc.\n"
        "CÁC QUY TẮC KHÁC:\n"
        "- CHỈ sửa lỗi chính tả, lỗi dấu thanh, lỗi gõ phím.\n"
        "- TUYỆT ĐỐI KHÔNG được tóm tắt, diễn giải, hay thay đổi từ ngữ.\n"
        "- Nếu văn bản gốc đã đúng, hãy lặp lại y hệt."
    )
    user_prompt = (
        "Dưới đây là một vài ví dụ:\n"
        f"{fs_examples}\n\n"
        "Bây giờ, hãy sửa văn bản sau đây. Chỉ trả về duy nhất văn bản đã được sửa.\n"
        f"Văn bản gốc: \"{original_text}\""
    )
    return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST] Văn bản đã sửa: \""

def generate_from_model(prompt, original_text_for_token_limit):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH)
    try: inputs = {k: v.to(model_device) for k, v in inputs.items()}
    except Exception: pass
    input_token_length = len(tokenizer.encode(original_text_for_token_limit, add_special_tokens=False))
    dynamic_min_new_tokens = int(input_token_length * 0.8)
    dynamic_max_new_tokens = int(input_token_length * 1.5) + 20
    gen_cfg = GenerationConfig(
        min_new_tokens=dynamic_min_new_tokens, max_new_tokens=dynamic_max_new_tokens,
        do_sample=False, num_beams=3, repetition_penalty=1.1, early_stopping=False,
        pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id
    )
    with torch.no_grad(): out = model.generate(**inputs, generation_config=gen_cfg)
    full_decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    if "[/INST]" in full_decoded:
        response_part = full_decoded.split("[/INST]")[-1].strip()
        if response_part.lower().startswith("văn bản đã sửa:"):
            response_part = response_part[len("Văn bản đã sửa:"):].strip()
        corrected = response_part.split('\n')[0].strip()
        corrected = re.sub(r'^["\']|["\']$', '', corrected)
        if corrected.endswith('"'): corrected = corrected[:-1]
        return corrected
    return ""

# ---------- Main correction function (HYBRID LOGIC) ----------
def correct_vietnamese_spelling(text, row_id, col_name, debug_print=False):
    original_text = "" if pd.isna(text) else str(text).strip()
    if not original_text: return text

    # *** BƯỚC 1: ÁP DỤNG SỬA LỖI RULE-BASED TRƯỚC ***
    pre_corrected_text = rule_based_correction(original_text)
    
    # Chỉ gọi LLM nếu văn bản vẫn còn khả năng có lỗi
    # Hoặc nếu bạn muốn LLM luôn kiểm tra lại, hãy bỏ điều kiện if này
    text_for_llm = pre_corrected_text
    
    # Sử dụng văn bản gốc làm key cache để tránh xử lý lại
    cache_key = (col_name, original_text)
    if cache_key in correction_cache: return correction_cache[cache_key]
    
    # *** BƯỚC 2: SỬA LỖI NÂNG CAO BẰNG LLM ***
    prompt = make_correction_prompt_llama2(text_for_llm)
    llm_corrected_text = generate_from_model(prompt, original_text_for_token_limit=text_for_llm)
    
    # Xác định kết quả cuối cùng cần kiểm tra
    # Nếu LLM không trả về gì hoặc trả về y hệt, kết quả là từ rule-based
    if not llm_corrected_text or llm_corrected_text == text_for_llm:
        final_corrected_text = pre_corrected_text
    else:
        final_corrected_text = llm_corrected_text
        
    # *** BƯỚC 3: KIỂM ĐỊNH CHẤT LƯỢNG SO VỚI BẢN GỐC ***
    accepted = False
    reason = "not_changed"
    final_text = original_text # Mặc định trả về bản gốc
    
    if final_corrected_text != original_text:
        current_base_threshold = STRICT_BASE_SIMILARITY if col_name in ['prompt', 'response'] else LENIENT_BASE_SIMILARITY
        orig_words = normalize_spaces(original_text).split()
        corr_words = normalize_spaces(final_corrected_text).split()
        orig_base = remove_diacritics_and_punct(original_text)
        corr_base = remove_diacritics_and_punct(final_corrected_text)
        base_sim = string_similarity(orig_base, corr_base)

        if abs(len(orig_words) - len(corr_words)) > WORD_COUNT_TOLERANCE:
            reason = f"word_count_mismatch_{len(orig_words)}_vs_{len(corr_words)}"
        elif base_sim < current_base_threshold:
            reason = f"base_similarity_too_low_{base_sim:.2f}_(threshold:{current_base_threshold})"
        elif len(original_text) > 0 and abs(len(final_corrected_text) - len(original_text)) / len(original_text) > LENGTH_CHANGE_ALLOWED_RATIO:
            reason = "length_changed_too_much"
        elif string_similarity(original_text, final_corrected_text) < ACCEPT_SIMILARITY_THRESHOLD:
            reason = f"low_similarity_{string_similarity(original_text, final_corrected_text):.2f}"
        elif original_text.endswith('?') and not final_corrected_text.endswith('?'):
            reason = "question_mark_removed"
        else:
            accepted = True
            reason = "accepted_change"
            final_text = final_corrected_text
            
    if debug_print:
        print("\n--- DEBUG SAMPLE ---")
        print(f"ID: {row_id} COLUMN: {col_name}")
        print(f"ORIGINAL: \n{original_text}")
        print(f"\nPRE-CORRECTED (Rule-based): \n{pre_corrected_text}") # Thêm dòng này để debug
        print(f"\nCORRECTED (LLM output): \n{llm_corrected_text}")
        print(f"\nRESULT: {'ACCEPTED' if accepted else 'REJECTED'}")
        print(f"REASON: {reason}")
        if final_corrected_text != original_text: print(f"BASE SIMILARITY: {string_similarity(remove_diacritics_and_punct(original_text), remove_diacritics_and_punct(final_corrected_text)):.4f}")
        print(f"FINAL TEXT: \n{final_text}")
        
    try:
        df_row = pd.DataFrame([{"id": row_id, "column": col_name, "original": original_text, "corrected_model": final_corrected_text, "final_text": final_text, "accepted": accepted, "reason": reason, "similarity": string_similarity(original_text, final_corrected_text), "base_similarity": string_similarity(remove_diacritics_and_punct(original_text), remove_diacritics_and_punct(final_corrected_text))}])
        log_header = not os.path.exists(DEBUG_LOG_CSV)
        df_row.to_csv(DEBUG_LOG_CSV, index=False, mode='a', header=log_header, encoding="utf-8-sig")
    except Exception as e:
        if debug_print: print(f"Warning: cannot write debug csv: {e}")
        
    correction_cache[cache_key] = final_text
    return final_text

# ---------------- Main loop ----------------
if not os.path.exists(INPUT_CSV): raise FileNotFoundError(f"Không tìm thấy file '{INPUT_CSV}'.")
df = pd.read_csv(INPUT_CSV)
df_global = df
corrected_ids = set()
if os.path.exists(DEBUG_LOG_CSV): os.remove(DEBUG_LOG_CSV)
try:
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
        current_id = row.get('id', index)
        for col in COLUMNS_TO_FIX:
            if col not in row or pd.isna(row[col]): continue
            original_text = row[col]
            debug_flag = (index < SAMPLE_DEBUG_N)
            corrected_text = correct_vietnamese_spelling(original_text, current_id, col, debug_print=debug_flag)
            if str(original_text) != str(corrected_text):
                corrected_ids.add(current_id)
                df.at[index, col] = corrected_text
except Exception as e: print(f"\nAn error occurred: {e}")
finally:
    print(f"\nĐã xử lý xong. Có {len(corrected_ids)} dòng được thay đổi.")
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nOutput đã được lưu vào {OUTPUT_CSV}")
    print(f"Log chi tiết đã được lưu vào {DEBUG_LOG_CSV}")