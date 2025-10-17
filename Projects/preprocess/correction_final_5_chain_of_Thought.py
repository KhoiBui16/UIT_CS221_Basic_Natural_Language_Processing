# fix_vietnamese_local_strict_spelling_only_fixed.py
# Final Version (v6): Chain-of-Thought Prompting to enhance reasoning and reduce hallucination.

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
OUTPUT_CSV = "./fixed-vihallu-public-test_final_5.csv"
COLUMNS_TO_FIX = ['context', 'prompt', 'response']

# *** UPDATE: Few-shot examples được cấu trúc lại theo dạng Chain-of-Thought ***
FEW_SHOT = [
    (
        "Ý nhĩa cũa viẹc tổ chưc cuôc thi Intervision là gì z?",
        """1. **Phân tích**:
- 'nhĩa': sai dấu hỏi/ngã -> 'nghĩa'
- 'cũa': sai dấu hỏi/ngã -> 'của'
- 'viẹc': sai dấu nặng -> 'việc'
- 'chưc': thiếu ký tự -> 'chức'
- 'cuôc': sai dấu thanh -> 'cuộc'
- 'z?': teencode -> 'vậy?' (hoặc bỏ đi)
2. **Lý giải**: Các từ trên sai lỗi chính tả và dấu câu cơ bản trong tiếng Việt.
3. **Sửa lỗi**: Ý nghĩa của việc tổ chức cuộc thi Intervision là gì?"""
    ),
    (
        "Nguồn thhu chinh của thuê truc tiếp la nhủg ai?",
        """1. **Phân tích**:
- 'thhu': thừa ký tự -> 'thu'
- 'chinh': sai dấu sắc/hỏi -> 'chính'
- 'thuê': sai dấu -> 'thuế'
- 'truc': thiếu dấu -> 'trực'
- 'la': thiếu dấu -> 'là'
- 'nhủg': sai dấu hỏi/ngã -> 'những'
2. **Lý giải**: Các từ trên sai lỗi gõ phím và dấu thanh.
3. **Sửa lỗi**: Nguồn thu chính của thuế trực tiếp là những ai?"""
    )
]

MAX_INPUT_LENGTH = 2048

# Giữ nguyên các ngưỡng lọc khắt khe của Ver 5
WORD_COUNT_TOLERANCE = 1
ACCEPT_SIMILARITY_THRESHOLD = 0.92
LENGTH_CHANGE_ALLOWED_RATIO = 0.15
STRICT_BASE_SIMILARITY = 0.98
LENIENT_BASE_SIMILARITY = 0.95

# debug outputs
correction_cache = {}
DEBUG_LOG_CSV = "debug_corrections_final_5.csv"
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

# ---------- Prompt & Generation ----------
def make_correction_prompt_llama2(original_text):
    """
    *** UPDATE: System prompt được thay đổi để yêu cầu model suy nghĩ từng bước (Chain-of-Thought) ***
    """
    fs_examples = "\n\n".join([f"Văn bản gốc: \"{o}\"\n{c}" for o, c in FEW_SHOT])
    
    system_prompt = (
        "Bạn là một trợ lý ngôn ngữ cẩn thận và có phương pháp. Hãy thực hiện chính xác 3 bước sau:\n"
        "1. **Phân tích**: Đọc kỹ 'Văn bản gốc' và liệt kê tất cả các lỗi chính tả, lỗi gõ phím, hoặc lỗi viết tắt.\n"
        "2. **Lý giải**: Với mỗi lỗi, giải thích ngắn gọn tại sao nó sai.\n"
        "3. **Sửa lỗi**: Dựa trên phân tích, viết lại câu hoàn chỉnh và đúng đắn.\n\n"
        "**QUY TẮC TUYỆT ĐỐI:** Chỉ sửa những gì bạn đã xác định là lỗi ở bước 1. KHÔNG được thay đổi từ ngữ hay ý nghĩa gốc."
    )

    user_prompt = (
        "Dưới đây là các ví dụ mẫu:\n"
        f"{fs_examples}\n\n"
        "--- BẮT ĐẦU NHIỆM VỤ ---\n"
        f"Văn bản gốc: \"{original_text}\""
    )
    
    return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"

def generate_from_model(prompt, original_text_for_token_limit):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH)
    try: inputs = {k: v.to(model_device) for k, v in inputs.items()}
    except Exception: pass
    
    input_token_length = len(tokenizer.encode(original_text_for_token_limit, add_special_tokens=False))
    
    # Cho phép model sinh output dài hơn để chứa chuỗi tư duy
    dynamic_min_new_tokens = int(input_token_length * 0.8)
    dynamic_max_new_tokens = int(input_token_length * 2.5) + 50 # Tăng không gian cho CoT

    gen_cfg = GenerationConfig(
        min_new_tokens=dynamic_min_new_tokens, max_new_tokens=dynamic_max_new_tokens,
        do_sample=False, num_beams=3, repetition_penalty=1.1, early_stopping=False,
        pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id
    )
    
    with torch.no_grad(): out = model.generate(**inputs, generation_config=gen_cfg)
    
    full_decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    
    if "[/INST]" in full_decoded:
        response_part = full_decoded.split("[/INST]")[-1].strip()
        
        # *** UPDATE: Logic trích xuất kết quả cuối cùng từ output CoT ***
        # Tìm dòng cuối cùng có chứa "Sửa lỗi:"
        lines = response_part.split('\n')
        corrected_line = ""
        for line in reversed(lines):
            # Tìm dòng bắt đầu bằng "3. Sửa lỗi:" hoặc chỉ "Sửa lỗi:"
            if "sửa lỗi:" in line.lower():
                corrected_line = line.split(":", 1)[-1].strip()
                break
        
        # Nếu không tìm thấy, lấy dòng cuối cùng không rỗng làm phương án dự phòng
        if not corrected_line:
            for line in reversed(lines):
                if line.strip():
                    corrected_line = line.strip()
                    break
                    
        corrected = re.sub(r'^["\']|["\']$', '', corrected_line)
        if corrected.endswith('"'): corrected = corrected[:-1]
        return corrected
        
    return ""

# ---------- Main correction function ----------
# (Không cần thay đổi logic ở đây, vì nó chỉ nhận kết quả cuối cùng)
def correct_vietnamese_spelling(text, row_id, col_name, debug_print=False):
    original_text = "" if pd.isna(text) else str(text).strip()
    if not original_text: return text
    
    cache_key = (col_name, original_text)
    if cache_key in correction_cache: return correction_cache[cache_key]
    
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
        print(f"\nCORRECTED (model returned): \n{corrected_text}") # Vẫn in ra để debug
        print(f"\nRESULT: {'ACCEPTED' if accepted else 'REJECTED'}")
        print(f"REASON: {reason}")
        if corrected_text and corrected_text != original_text: print(f"BASE SIMILARITY: {base_sim:.4f}")
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
    
    print("\n====================================")
    print(f"\nDanh sách những id thay đổi:")
    for idx in corrected_ids:
        print(f"id: {idx}")