# fix_vietnamese_local_strict_spelling_only_fixed.py
# Fixed: New robust prompt and stricter, simpler validation logic.

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
OUTPUT_CSV = "./fixed-vihallu-public-test_final_1.csv"
COLUMNS_TO_FIX = ['context', 'prompt', 'response'] # Bật lại context để kiểm tra

FEW_SHOT = [
    ("Toi6 dang hoc online.", "Tôi đang học online."),
    ("Ý nhĩa cũa viẹc tổ chưc cuôc thi Intervision là gì z?", "Ý nghĩa của việc tổ chức cuộc thi Intervision là gì?"),
    ("Chiec xe chay rât nhanh", "Chiếc xe chạy rất nhanh"),
    ("Lam on cho biet thong tin", "Làm ơn cho biết thông tin")
]


MAX_INPUT_LENGTH = 1024
# MAX_NEW_TOKENS_CORRECTION đã bị loại bỏ, thay bằng logic linh hoạt

# Các ngưỡng này sẽ được sử dụng trong logic lọc mới
ACCEPT_SIMILARITY_THRESHOLD = 0.85 # Ngưỡng Levenshtein tối thiểu để chấp nhận một sự thay đổi
BASE_SIMILARITY_THRESHOLD = 0.95 # *** MỚI: Ngưỡng tương đồng cho chuỗi không dấu, phải rất cao
LENGTH_CHANGE_ALLOWED_RATIO = 0.1 # Tăng nhẹ lên 10% để linh hoạt hơn

# debug outputs
correction_cache = {}
DEBUG_LOG_CSV = "debug_corrections_final_1.csv"
SAMPLE_DEBUG_N = 20  # print console debug for first N samples (set 0 to disable)

# ---------------- device info ----------------
print("CUDA available:", torch.cuda.is_available())

# print thresholds for debugging
print("\n=== DEBUG: Thresholds ===")
print("MAX_INPUT_LENGTH:", MAX_INPUT_LENGTH)
print("ACCEPT_SIMILARITY_THRESHOLD:", ACCEPT_SIMILARITY_THRESHOLD)
print("BASE_SIMILARITY_THRESHOLD:", BASE_SIMILARITY_THRESHOLD)
print("LENGTH_CHANGE_ALLOWED_RATIO:", LENGTH_CHANGE_ALLOWED_RATIO)
print("DEBUG_LOG_CSV:", DEBUG_LOG_CSV, " SAMPLE_DEBUG_N:", SAMPLE_DEBUG_N)
print("====================================\n")

# Ctrl+C handler - Rất hữu ích để lưu tiến trình
df_global = None # Biến toàn cục để lưu dataframe
def signal_handler(sig, frame):
    print("\n\n[Ctrl+C] Đang lưu tiến trình...")
    if df_global is not None:
        df_global.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"Đã lưu thành công vào {OUTPUT_CSV}")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


# ---------- tokenizer/model setup ----------
tokenizer_kwargs = {"use_fast": False, "token": HF_TOKEN, "padding_side": "right"}
model_kwargs = {"token": HF_TOKEN}

def load_tokenizer_local_or_remote(model_id):
    try:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True, **{k: v for k, v in tokenizer_kwargs.items() if k != "token"})
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
    
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if tok.bos_token_id is None:
        tok.bos_token_id = 1
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
            model_path = snapshot_download(model_id, token=HF_TOKEN, cache_dir=cache_dir) if HF_TOKEN else cache_dir
            return m, model_path
        except Exception as e:
            last_e = e
            continue
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
    """ *** UPDATED: Bỏ cả dấu câu để so sánh từ gốc sạch hơn *** """
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = unicodedata.normalize('NFC', s)
    s = re.sub(r'[^\w\s]', '', s) # Bỏ tất cả các ký tự không phải chữ, số, hoặc khoảng trắng
    return s

def normalize_spaces(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def string_similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

# ---------- Prompt & Generation ----------

def make_correction_prompt(original_text):
    """
    Prompt mới, mạnh mẽ hơn theo dạng Role-Playing và Instruction.
    """
    fs_examples = "\n".join([f"Văn bản gốc: \"{o}\"\nVăn bản đã sửa: \"{c}\"" for o, c in FEW_SHOT])

    prompt = (
        "Bạn là một chuyên gia sửa lỗi chính tả và ngữ pháp tiếng Việt CỰC KỲ CẨN THẬN.\n"
        "Nhiệm vụ của bạn là đọc 'Văn bản gốc' và chỉ sửa những lỗi sau:\n"
        "- Lỗi chính tả (ví dụ: 'chuyện' thay vì 'chụyện').\n"
        "- Lỗi dấu thanh (ví dụ: 'hoà' thay vì 'hòa').\n"
        "- Lỗi gõ phím (ví dụ: 'onlien' -> 'online', 'toi6' -> 'tôi').\n"
        "- Khoảng trắng thừa.\n\n"
        "--- CÁC QUY TẮC TUYỆT ĐỐI KHÔNG ĐƯỢC VI PHẠM ---\n"
        "1. **KHÔNG** thay đổi bất kỳ từ nào, chỉ sửa lỗi trên từ đó.\n"
        "2. **KHÔNG** thêm hoặc xóa từ. SỐ LƯỢNG TỪ PHẢI GIỮ NGUYÊN.\n"
        "3. **KHÔNG** thay đổi trật tự các từ trong câu.\n"
        "4. **KHÔNG** thay đổi ý nghĩa của câu.\n"
        "5. **KHÔNG** thay đổi các con số hoặc tên riêng.\n"
        "6. **KHÔNG** thay đổi loại câu (câu hỏi vẫn phải kết thúc bằng dấu chấm hỏi).\n"
        "7. Nếu văn bản gốc đã đúng, hãy lặp lại chính xác nó.\n\n"
        "--- VÍ DỤ ---\n"
        f"{fs_examples}\n\n"
        "--- BẮT ĐẦU NHIỆM VỤ ---\n"
        f"Văn bản gốc: \"{original_text}\"\n"
        "Văn bản đã sửa:"
    )
    return prompt

def generate_from_model(prompt, original_text_for_token_limit):
    """
    *** UPDATED: Thêm tham số để giới hạn token sinh ra một cách linh hoạt ***
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH)

    # --- ĐÂY LÀ THAY ĐỔI QUAN TRỌNG NHẤT ---
    # Tính toán số token tối đa cần sinh ra dựa trên độ dài của văn bản gốc
    # Cho phép dài hơn 20% so với bản gốc + 10 token đệm để xử lý các trường hợp đặc biệt
    input_token_length = len(tokenizer.encode(original_text_for_token_limit))
    dynamic_max_new_tokens = int(input_token_length * 1.2) + 10
    # --- KẾT THÚC THAY ĐỔI ---
    
    try:
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
    except Exception:
        pass
    
    gen_cfg = GenerationConfig(
        max_new_tokens=dynamic_max_new_tokens, # Sử dụng giá trị linh hoạt
        do_sample=False,
        num_beams=3, # Tăng lên 3 để có kết quả tốt hơn greedy search
        early_stopping=True, # Dừng sớm khi tìm thấy chuỗi tốt
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg)
    
    full_decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    
    # Trích xuất phần trả lời một cách an toàn
    if "Văn bản đã sửa:" in full_decoded:
        corrected_part = full_decoded.rsplit("Văn bản đã sửa:", 1)[1]
        corrected = corrected_part.strip().split('\n')[0].strip()
        corrected = re.sub(r'^["\']|["\']$', '', corrected)
        return corrected
    else:
        return ""


# ---------- Main correction function ----------

def correct_vietnamese_spelling(text, row_id, col_name, debug_print=False):
    original_text = "" if pd.isna(text) else str(text).strip()
    if not original_text:
        return text

    cache_key = (col_name, original_text)
    if cache_key in correction_cache:
        return correction_cache[cache_key]

    prompt = make_correction_prompt(original_text)
    corrected_text = generate_from_model(prompt, original_text_for_token_limit=original_text)

    # --- Logic lọc và xác thực mới ---
    accepted = False
    reason = "not_changed"
    final_text = original_text # Mặc định là giữ nguyên

    base_sim = 0.0
    if corrected_text and corrected_text != original_text:
        # *** LOGIC LỌC ĐÃ ĐƯỢC CẬP NHẬT ***
        orig_base = remove_diacritics_and_punct(original_text)
        corr_base = remove_diacritics_and_punct(corrected_text)
        base_sim = string_similarity(orig_base, corr_base)

        # Quy tắc 1: Độ tương đồng của chuỗi không dấu phải RẤT CAO
        if base_sim < BASE_SIMILARITY_THRESHOLD:
            reason = f"base_similarity_too_low_{base_sim:.2f}"
        # Quy tắc 2: Độ tương đồng của chuỗi gốc phải đủ cao
        elif string_similarity(original_text, corrected_text) < ACCEPT_SIMILARITY_THRESHOLD:
            reason = f"low_similarity_{string_similarity(original_text, corrected_text):.2f}"
        # Quy tắc 3: Thay đổi độ dài không quá lớn
        elif len(original_text) > 0 and abs(len(corrected_text) - len(original_text)) / len(original_text) > LENGTH_CHANGE_ALLOWED_RATIO:
             reason = "length_changed_too_much"
        # Quy tắc 4: Câu hỏi phải giữ lại dấu '?'
        elif original_text.endswith('?') and not corrected_text.endswith('?'):
            reason = "question_mark_removed"
        else:
            accepted = True
            reason = "accepted_change"
            final_text = corrected_text

    # Log và debug
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
        df_row = pd.DataFrame([{
            "id": row_id,
            "column": col_name,
            "original": original_text,
            "corrected_model": corrected_text,
            "final_text": final_text,
            "accepted": accepted,
            "reason": reason,
            "similarity": string_similarity(original_text, corrected_text) if corrected_text else 1.0,
            "base_similarity": base_sim
        }])
        log_header = not os.path.exists(DEBUG_LOG_CSV)
        df_row.to_csv(DEBUG_LOG_CSV, index=False, mode='a', header=log_header, encoding="utf-8-sig")
    except Exception as e:
        if debug_print:
            print(f"Warning: cannot write debug csv: {e}")

    correction_cache[cache_key] = final_text
    return final_text

# ---------------- Main loop ----------------
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Không tìm thấy file '{INPUT_CSV}'.")

df = pd.read_csv(INPUT_CSV)
df_global = df # Gán vào biến toàn cục để Ctrl+C handler có thể truy cập
df_orig = df.copy()

corrected_ids = set()

# Xóa file log cũ nếu có
if os.path.exists(DEBUG_LOG_CSV):
    os.remove(DEBUG_LOG_CSV)

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
print(f"Log chi tiết đã được lưu vào {DEBUG_LOG_CSV}")


print("\n====================================")
print(f"\nDanh sách những id thay đổi:")
for idx in corrected_ids:
    print(f"id: {idx}")

