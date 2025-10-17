# fix_vietnamese_local_strict_spelling_only_fixed.py
# Final Polished Version: Tích hợp hàm trích xuất danh từ riêng và số thông minh hơn.

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

public_test_correction_folder = "./vihallu_public_test_correction"
os.makedirs(public_test_correction_folder, exist_ok=True)

# ---------------- CONFIG ----------------
MODEL = "VietnamAIHub/Vietnamese_llama2_7B_8K_SFT_General_domain"
INPUT_CSV = "./vihallu-public-test.csv"
OUTPUT_CSV =  os.path.join(public_test_correction_folder, "./fixed-vihallu-public-test_final_v8_temp.csv")
COLUMNS_TO_FIX = ['context', 'prompt', 'response']

FEW_SHOTS = [
    ("Nguồn thhu chinh của thuê truc tiếp la nhủg ai?", "Nguồn thu chính của thuế trực tiếp là những ai?"),
    ("Ý nhĩa cũa viẹc tổ chưc cuôc thi Intervision là gì z?", "Ý nghĩa của việc tổ chức cuộc thi Intervision là gì?"),
    ("thủ đô ha nội là trung tâm văn hóa", "thủ đô Hà Nội là trung tâm văn hóa"),
    ("...không đứng đầu thế giới về dự trữ ngoại tệ?", "...không đứng đầu thế giới về dự trữ ngoại tệ?"),
    ("Các nhà khoa học muốn tìm đến S.J thì phải đến đâu?", "Các nhà khoa học muốn tìm đến S.J thì phải đến đâu?"),
    ("...cuộc thi hát nổi tiếng thường niên...", "...cuộc thi hát nổi tiếng thường niên..."),
]

MAX_INPUT_LENGTH = 1700
WORD_COUNT_TOLERANCE = 1
ACCEPT_SIMILARITY_THRESHOLD = 0.88
LENGTH_CHANGE_ALLOWED_RATIO = 0.15 
STRICT_BASE_SIMILARITY      = 0.97
LENIENT_BASE_SIMILARITY     = 0.94      

debug_correction_villua_folder = "./vihallu_public_correction"
os.makedirs(debug_correction_villua_folder, exist_ok=True)
DEBUG_LOG_CSV =  os.path.join(debug_correction_villua_folder, "debug_corrections_final_v8_temp.csv")
SAMPLE_DEBUG_N = 20


# ---------------- device info ----------------
print("CUDA available:", torch.cuda.is_available())
print("\n=== DEBUG: Thresholds (Final Optimized) ===")
print(f"STRICT (prompt, response): BASE_SIMILARITY >= {STRICT_BASE_SIMILARITY}")
print(f"LENIENT (context): BASE_SIMILARITY >= {LENIENT_BASE_SIMILARITY}")
print(f"GENERAL: ACCEPT_SIMILARITY >= {ACCEPT_SIMILARITY_THRESHOLD}")
print("====================================\n")

# Ctrl+C handler
df_global = None
def signal_handler(sig, frame):
    print("\n\n[Ctrl+C] Saving progress...")
    if df_global is not None:
        df_global.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"Successfully saved to {OUTPUT_CSV}")
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
    print(f"Cache huggingface path: {cache_dir}")
    loading_strategies = [
        {"name": "8bit", "kwargs": {"quantization_config": BitsAndBytesConfig(load_in_8bit=True), "device_map": "auto", "token": HF_TOKEN, "cache_dir": cache_dir}},
        {"name": "bfloat16", "kwargs": {"torch_dtype": torch.bfloat16, "device_map": "auto", "token": HF_TOKEN, "cache_dir": cache_dir}},
    ]
    
    last_e = None
    for strat in loading_strategies:    
        try:
            print(f"Trying to load with config: {strat['name']}")
            m = AutoModelForCausalLM.from_pretrained(model_id, **strat["kwargs"])
            print(f"Successfully loaded with config: {strat['name']}")
            model_path = (
                snapshot_download(model_id, token=HF_TOKEN, cache_dir=cache_dir)
                if HF_TOKEN
                else cache_dir
            )
            return m, model_path
        except Exception as e:
            print(f"Failed with config: {strat['name']} - {type(e).__name__}: {e}")
            last_e = e
    raise RuntimeError("Cannot load model") from last_e


print("Loading tokenizer...")
tokenizer = load_tokenizer_local_or_remote(MODEL)

print("Loading model...")
model, model_source = load_model_local_or_snapshot(MODEL)
model.eval()

try: 
    model_device = next(model.parameters()).device
except: 
    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Model loaded from:", model_source)

# ---------- Utilities ----------
def remove_diacritics_and_punct(s: str):
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = unicodedata.normalize('NFC', s)
    s = re.sub(r'[^\w\s]', '', s)
    return s.lower()

def normalize_spaces(s: str): 
    return re.sub(r'\s+', ' ', s).strip()

def string_similarity(a, b): 
    return difflib.SequenceMatcher(None, a, b).ratio()

# *** UPDATE: HÀM TRÍCH XUẤT SỐ ĐƯỢC NÂNG CẤP ***
def extract_numbers(text: str):
    """Trích xuất tất cả các chuỗi số, bao gồm cả số có dấu phẩy và dấu chấm."""
    return set(re.findall(r'\d+[,.]?\d*', text))

def extract_proper_nouns(text: str):
    """
    Trích xuất các cụm danh từ riêng tiềm năng (chuỗi các từ viết hoa liền nhau).
    Hỗ trợ đầy đủ ký tự tiếng Việt và các trường hợp có dấu '.' hoặc '-'.
    """
    vietnamese_uppercase = "A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ"
    pattern = r'\b([' + vietnamese_uppercase + r'][\w\.-]+' + r'(?:\s+[' + vietnamese_uppercase + r'][\w\.-]+)*)\b'
    found_nouns = set(re.findall(pattern, text))
    common_acronyms = {'AI', 'LLM', 'UIT', 'SED', 'SKLP', 'GDP', 'UNESCO'}
    return {noun for noun in found_nouns if noun not in common_acronyms}


# ---------- Prompt & Generation ----------
def make_correction_prompt_llama2(original_text):
    fs_examples = "\n".join([f"Văn bản gốc: \"{o}\"\nVăn bản đã sửa: \"{c}\"" for o, c in FEW_SHOTS])
    system_prompt = (
        "Bạn là một robot hiệu đính MÁY MÓC và CẨN TRỌNG. "
        "**QUY TẮC SỐ 1 (QUAN TRỌNG NHẤT):** KHÔNG THAY ĐỔI Ý NGHĨA. Giữ nguyên từ đúng ngữ nghĩa (ví dụ: 'ngoại tệ'), giữ nguyên từ viết tắt không rõ (ví dụ: 'S.J'). "
        "**QUY TẮC SỐ 2:** CHỈ SỬA lỗi chính tả, lỗi gõ phím, lỗi dấu thanh, lỗi viết hoa. "
        "**QUY TẮC SỐ 3:** Nếu văn bản đã đúng, lặp lại y hệt."
    )
    user_prompt = (
        "Các ví dụ:\n"
        f"{fs_examples}\n\n"
        "Sửa văn bản sau. Chỉ trả về văn bản đã sửa.\n"
        f"Văn bản gốc: \"{original_text}\""
    )
    return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST] Văn bản đã sửa: \""

def generate_from_model(prompt, original_text_for_token_limit):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH)
    try: inputs = {k: v.to(model_device) for k, v in inputs.items()}
    except Exception: pass
    
    input_token_length = len(tokenizer.encode(original_text_for_token_limit, add_special_tokens=False))
    
    dynamic_min_new_tokens = int(input_token_length * 0.8)
    dynamic_max_new_tokens = int(input_token_length * 1.5) + 50
    
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

correction_cache = {}

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
        passes_initial_checks = False
        
        current_base_threshold = STRICT_BASE_SIMILARITY if col_name in ['prompt', 'response'] else LENIENT_BASE_SIMILARITY
        orig_words = normalize_spaces(original_text).split()
        corr_words = normalize_spaces(corrected_text).split()
        
        base_sim = string_similarity(remove_diacritics_and_punct(original_text), remove_diacritics_and_punct(corrected_text))
        
        if abs(len(orig_words) - len(corr_words)) > WORD_COUNT_TOLERANCE:
            reason = f"word_count_mismatch: original_words:{len(orig_words)} vs correction_word: {len(corr_words)}"
        elif base_sim < current_base_threshold:
            reason = f"base_similarity_too_low_{base_sim:.2f}_(threshold:{current_base_threshold})"
        elif len(original_text) > 0 and abs(len(corrected_text) - len(original_text)) / len(original_text) > LENGTH_CHANGE_ALLOWED_RATIO:
            reason = "length_changed_too_much"
        elif string_similarity(original_text, corrected_text) < ACCEPT_SIMILARITY_THRESHOLD:
            reason = f"low_similarity_{string_similarity(original_text, corrected_text):.2f}"
        elif original_text.endswith('?') and not corrected_text.endswith('?'):
            reason = "question_mark_removed"
        else:
            passes_initial_checks = True
            
        if passes_initial_checks:
            original_numbers = extract_numbers(original_text)
            corrected_numbers = extract_numbers(corrected_text)
            original_nouns = extract_proper_nouns(original_text)
            corrected_nouns = extract_proper_nouns(corrected_text)
            
            # Kiểm tra sự tương đương tuyệt đối của các "sự thật"
            if original_numbers != corrected_numbers:
                reason = f"numbers_altered: {original_numbers.symmetric_difference(corrected_numbers)}"
                passes_initial_checks = False
            elif original_nouns != corrected_nouns:
                reason = f"proper_nouns_altered: {original_nouns.symmetric_difference(corrected_nouns)}"
                passes_initial_checks = False

        if passes_initial_checks:
            accepted = True
            reason = "accepted_change"
            final_text = corrected_text
            
    if debug_print:
        print("\n--- DEBUG SAMPLE ---")
        print(f"ID: {row_id} | COLUMN: {col_name}")
        print(f"+ ORIGINAL: \n{original_text}")
        print(f"\n+ CORRECTED (model returned): \n{corrected_text}")
        
        if corrected_text and corrected_text != original_text:
            print("\n--- METRICS & THRESHOLDS ---")
            print(f"  - Base Similarity: {base_sim:.4f}")
            print(f"  - Direct Similarity: {string_similarity(original_text, corrected_text):.4f}")
            
            print("\n--- FACT CHECKING ---")
            print(f"  - Original Numbers (len: {len(extract_numbers(original_text))}): \n{extract_numbers(original_text)}")
            print(f"  - Corrected Numbers (len: {extract_numbers(corrected_text)}): \n{extract_numbers(corrected_text)}")
            print(f"  - \nOriginal Nouns (len: {len(extract_proper_nouns(original_text))}): \n{extract_proper_nouns(original_text)}")
            print(f"  - Corrected Nouns (len: {extract_proper_nouns(corrected_text)}): \n{extract_proper_nouns(corrected_text)}")
        else:
            print("\n--- METRICS & THRESHOLDS ---")
            print("No change proposed by model.")

        print("\n--- DECISION ---")
        print(f"RESULT: {'ACCEPTED' if accepted else 'REJECTED'}")
        if not accepted:
            print(f"REASON: {reason}")
        print(f"FINAL TEXT: \n{final_text}")
        print("-" * 20)
        
    try:
        df_row = pd.DataFrame([{"id": row_id, "column": col_name, "original": original_text, "corrected_model": corrected_text, "final_text": final_text, "accepted": accepted, "reason": reason, "similarity": string_similarity(original_text, corrected_text) if corrected_text else 1.0, "base_similarity": base_sim}])
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

