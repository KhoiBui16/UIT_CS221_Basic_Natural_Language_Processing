#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fix_spelling.py (final)
- Generator: bmd1905/vietnamese-correction-v2 (seq2seq, max tokens = 512)
- Validator (NLI): joeddav/xlm-roberta-large-xnli (max tokens = 1024)
- Chỉ sửa 3 cột: context, prompt, response
- Giữ nguyên id và label; nếu input không có label thì tạo predict_label (empty).
- Xử lý theo hàng (mỗi hàng cùng lúc các cột cần sửa) để tránh sai thứ tự.
- Resume dựa trên DEBUG_LOG.

Thiết kế dựa trên EDA bạn cung cấp (thống kê số từ):
(*** Đây là phần suy luận / lựa chọn tham số dựa trên EDA của bạn ***)
- # test / # train summary (số từ):
    context mean ~179, 90% ~261, 99% ~401 (train max up to 1537)
    prompt mean ~26, 99% ~63
    response mean ~39, 99% ~68
    combined mean ~246, 99% up to ~486 (test) / 1630 (train max)

[Suy luận] Dựa trên EDA trên và giới hạn model:
- Generator seq2seq có max 512 tokens -> **không gộp 3 cột**; xử lý từng cột riêng.
- Prompt & response trung bình ngắn, hầu như an toàn với 512; context có nhiều outlier -> cần truncate by tokens cho context > 512.
- Validator (xnli) có max 1024 -> dùng 1024 khi gọi NLI (song vẫn truncate để an toàn).

Mục tiêu: giữ nguyên format CSV, đảm bảo mapping id chính xác, ghi log chi tiết (token lengths, lý do truncate), không để model tạo nội dung mới/chế thông tin.

"""

import os
import re
import sys
import unicodedata
import difflib
import signal
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from underthesea import ner
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from dotenv import load_dotenv

# ---------------- CONFIG ----------------
GENERATOR_MODEL_NAME = "bmd1905/vietnamese-correction-v2"
VALIDATOR_MODEL_NAME = "joeddav/xlm-roberta-large-xnli"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(ROOT_DIR, "..", "data", "vihallu-public-test.csv")
OUTPUT_CSV = os.path.join(ROOT_DIR, "..", "data", "fixed-vihallu-public-test-NLI-validated.csv")
DEBUG_LOG = os.path.join(ROOT_DIR, "..", "data", "debug_data_vi_correction_log.csv")

COLUMNS_TO_FIX = ['context', 'prompt', 'response']

# Hard constraints from your message:
# - generator seq2seq max = 512
# - xnli max = 1024
GENERATOR_MAX_TOKENS = 512
VALIDATOR_MAX_TOKENS = 512

# Batch & token settings (tuneable)
ROWS_BATCH_SIZE = 4            # default from you; adjust if OOM
MAX_PROMPTS_PER_GEN = 64
MAX_NEW_TOKENS = 256  # Tăng từ 128 để tránh truncation

# Reserve tokens for instruction/special tokens when composing seq2seq input
INSTRUCTION_RESERVE = 12  # [Suy luận] an toàn để dành cho prompt prefix + special tokens

# Thresholds (kept from your original; can be tuned)
ACCEPT_SIMILARITY_THRESHOLD = 0.7
LENGTH_CHANGE_ALLOWED_RATIO = 0.3
STRICT_BASE_SIMILARITY = 0.92
LENIENT_BASE_SIMILARITY = 0.85
WORD_COUNT_TOLERANCE = 50  # Tăng từ 15 lên 50 để cho phép truncation

# EDA-driven heuristics [Suy luận]
# - For `context` column (longest), allow up to CONTEXT_ALLOWED_TOKENS = GENERATOR_MAX_TOKENS - INSTRUCTION_RESERVE
# - For `prompt` and `response`, we allow more headroom though typical lengths are small
CONTEXT_ALLOWED_TOKENS = GENERATOR_MAX_TOKENS - INSTRUCTION_RESERVE
SHORT_COL_ALLOWED_TOKENS = min(256, GENERATOR_MAX_TOKENS - INSTRUCTION_RESERVE)

# ---------------- SETUP ----------------
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Tokenizers & Models ---
print(f"Loading Generator tokenizer: {GENERATOR_MODEL_NAME}...")
gen_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
print(f"Loading Generator model: {GENERATOR_MODEL_NAME}...")
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL_NAME)

print(f"Loading Validator tokenizer: {VALIDATOR_MODEL_NAME}...")
val_tokenizer = AutoTokenizer.from_pretrained(VALIDATOR_MODEL_NAME)
print(f"Loading Validator model: {VALIDATOR_MODEL_NAME}...")
val_model = AutoModelForSequenceClassification.from_pretrained(VALIDATOR_MODEL_NAME)

# Move to device (safe fallback to CPU if OOM)
try:
    gen_model.to(device)
    val_model.to(device)
    gen_model.eval()
    val_model.eval()
except RuntimeError as e:
    print("Warning: moving models to device failed (OOM?). Falling back to CPU. Error:", e)
    device = torch.device("cpu")
    gen_model.to(device)
    val_model.to(device)

# ---------------- UTILITIES ----------------

def remove_diacritics_and_punct(s: str) -> str:
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = unicodedata.normalize('NFC', s)
    s = re.sub(r"[^\w\s]", '', s)
    return s.lower()


def normalize_spaces(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()


def string_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def extract_numbers(text: str):
    return set(re.findall(r'\d+[,.]?\d*', text))


def extract_proper_nouns_ner(text: str):
    """
    Sử dụng underthesea NER để trích xuất các danh từ riêng (Tên người, Địa điểm, Tổ chức).
    Chính xác hơn nhiều so với phương pháp regex.
    """
    # Xử lý trường hợp input rỗng hoặc không phải chuỗi
    if not text or not isinstance(text, str):
        return set()

    proper_nouns = set()
    # Các loại thực thể chúng ta quan tâm: Person, Location, Organization
    target_labels = {'PER', 'LOC', 'ORG'}

    try:
        # Gọi mô hình NER của underthesea
        entities = ner(text)
        
        # Output của ner() là một list các tuple, ví dụ: ('Hồ Chí Minh', 'Np', 'B-PER')
        for entity_text, pos_tag, ner_label in entities:
            # ner_label có dạng 'B-TAG' hoặc 'I-TAG' (ví dụ: 'B-PER')
            # Chúng ta chỉ cần kiểm tra xem TAG có nằm trong target_labels không
            tag = ner_label.split('-')[-1]
            if tag in target_labels:
                proper_nouns.add(entity_text.strip())

    except Exception as e:
        # Phòng trường hợp underthesea gặp lỗi với một chuỗi văn bản nào đó
        print(f"Warning: underthesea NER failed for text snippet. Error: {e}")
        return set() # Trả về set rỗng nếu có lỗi

    # Vẫn giữ lại bộ lọc các từ viết tắt phổ biến
    common_acronyms = {'AI', 'LLM', 'UIT', 'GDP', 'UNESCO'}
    return {noun for noun in proper_nouns if noun not in common_acronyms}

# Token helpers

def tokens_len(text: str, tokenizer) -> int:
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        # fallback: approximate by splitting on spaces
        return len(text.split())
    
def _clean_model_output(text: str) -> str:
    """
    Loại bỏ các tiền tố instruction mà model có thể trả về, ví dụ:
    'Sửa chính tả:', 'Sửa chính tả', 'Correction:', ...
    Trả về chuỗi đã clean và strip().
    """
    if text is None:
        return ""
    s = str(text).strip()
    # xóa tiền tố tiếng Việt "Sửa chính tả:" (có thể có dấu hai chấm cả : hoặc )
    s = re.sub(r'^\s*Sửa chính tả\s*[:]\s*', '', s, flags=re.IGNORECASE)
    # xóa một vài tiền tố phổ biến tiếng Anh nếu model trả về
    s = re.sub(r'^\s*Correction\s*[:]\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^\s*Correct spelling\s*[:]\s*', '', s, flags=re.IGNORECASE)
    # remove repeated instruction if model echoes it multiple times
    s = re.sub(r'^\s*(Sửa chính tả\s*[:]\s*)+', '', s, flags=re.IGNORECASE)
    return s.strip()


def truncate_text_by_tokens(text: str, max_tokens: int, tokenizer, keep_head_ratio: float = 0.6):
    """Truncate by tokens; keep head and tail with a separator so important start/end preserved."""
    if not text:
        return text, False, 0
    toks = tokenizer.encode(text, add_special_tokens=False)
    if len(toks) <= max_tokens:
        return text, False, len(toks)
    head_k = int(max_tokens * keep_head_ratio)
    tail_k = max_tokens - head_k
    head_text = tokenizer.decode(toks[:head_k], skip_special_tokens=True)
    tail_text = tokenizer.decode(toks[-tail_k:], skip_special_tokens=True)
    truncated = (head_text.rstrip() + " ... " + tail_text.lstrip()).strip()
    return truncated, True, len(toks)

# Prompt maker for seq2seq generator

def make_correction_prompt(original_text: str) -> str:
    return f"Sửa chính tả: {original_text}"

# Generate wrapper (safe chunks, memory-aware)

def generate_from_model(prompts, max_new_tokens=MAX_NEW_TOKENS):
    if isinstance(prompts, str):
        prompts = [prompts]
    results = []
    for i in range(0, len(prompts), MAX_PROMPTS_PER_GEN):
        chunk = prompts[i:i+MAX_PROMPTS_PER_GEN]
        inputs = gen_tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=GENERATOR_MAX_TOKENS).to(device)
        try:
            with torch.no_grad():
                outputs = gen_model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=5, early_stopping=True)
            decoded = gen_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        except Exception as e:
            print(f"Generation chunk failed: {e}. Falling back to per-prompt generation.")
            decoded = []
            for p in chunk:
                try:
                    in_single = gen_tokenizer(p, return_tensors="pt", truncation=True, max_length=GENERATOR_MAX_TOKENS).to(device)
                    with torch.no_grad():
                        out = gen_model.generate(**in_single, max_new_tokens=max_new_tokens, num_beams=5, early_stopping=True)
                    decoded.append(gen_tokenizer.decode(out[0], skip_special_tokens=True))
                except Exception as e2:
                    print(f"Per-prompt generation failed: {e2}")
                    decoded.append("")
        results.extend(decoded)
        # free memory
        try:
            del inputs
            torch.cuda.empty_cache()
        except Exception:
            pass
    return results

# NLI helper

def verify_with_nli_label_prob(premise: str, hypothesis: str):
    if not premise or not hypothesis:
        return None, 0.0
    inputs = val_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True, max_length=VALIDATOR_MAX_TOKENS).to(device)
    with torch.no_grad():
        outputs = val_model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_id = int(torch.argmax(logits, dim=1).item())
    label = val_model.config.id2label[pred_id]
    prob = float(probs[pred_id])
    return label, prob

# Validate proposal

def validate_proposal_with_nli(original_text: str, proposal: str, col_name: str):
    original_text = "" if pd.isna(original_text) else str(original_text).strip()
    proposal = "" if pd.isna(proposal) else str(proposal).strip()

    base_sim = None
    sim_val = None
    len_ratio = None
    numbers_diff = ""
    nouns_diff = ""
    nli_label = None
    nli_prob = None

    if not original_text:
        return original_text, "reject", "empty_original", base_sim, sim_val, len_ratio, nli_label, nli_prob, numbers_diff, nouns_diff
    if not proposal:
        return original_text, "reject", "empty_proposal", base_sim, sim_val, len_ratio, nli_label, nli_prob, numbers_diff, nouns_diff
    if original_text.lower() == proposal.lower():
        # Kiểm tra xem có phải chỉ khác về dấu câu/khoảng trắng không
        if original_text != proposal:
            # Có sự khác biệt nhỏ về format -> chấp nhận
            reason = f"accept_minor_formatting (sim=1.0, base_sim=1.0, len_ratio=0.0)"
            return proposal, "accept", reason, 1.0, 1.0, 0.0, "entailment", 1.0, numbers_diff, nouns_diff
        else:
            return original_text, "reject", "identical (no change)", 1.0, 1.0, 0.0, None, 0.0, numbers_diff, nouns_diff

    base_sim = string_similarity(remove_diacritics_and_punct(original_text), remove_diacritics_and_punct(proposal))
    sim_val = string_similarity(original_text, proposal)
    len_ratio = abs(len(proposal) - len(original_text)) / max(1, len(original_text))

    orig_words = normalize_spaces(original_text).split()
    corr_words = normalize_spaces(proposal).split()
    current_base_threshold = STRICT_BASE_SIMILARITY if col_name in ['prompt', 'response'] else LENIENT_BASE_SIMILARITY

    word_diff = abs(len(orig_words) - len(corr_words))
    
    # Nếu proposal ngắn hơn đáng kể, có thể do truncation - kiểm tra xem có phải là prefix không
    if len(corr_words) < len(orig_words) and word_diff > WORD_COUNT_TOLERANCE:
        # Kiểm tra nếu proposal là prefix của original (do truncation)
        proposal_normalized = normalize_spaces(proposal).lower()
        original_normalized = normalize_spaces(original_text).lower()
        if original_normalized.startswith(proposal_normalized):
            # Đây có thể là truncation hợp lệ, chỉ từ chối nếu quá ngắn
            if len(corr_words) < len(orig_words) * 0.3:  # Nếu proposal < 30% original thì từ chối
                reason = f"proposal_too_short_truncation: original_words:{len(orig_words)} vs correction_words:{len(corr_words)} (proposal < 30% original)"
                return original_text, "reject", reason, base_sim, sim_val, len_ratio, nli_label, nli_prob, numbers_diff, nouns_diff
        else:
            # Không phải truncation, áp dụng rule cũ
            if word_diff > WORD_COUNT_TOLERANCE:
                reason = f"word_count_mismatch: original_words:{len(orig_words)} vs correction_words:{len(corr_words)}"
                return original_text, "reject", reason, base_sim, sim_val, len_ratio, nli_label, nli_prob, numbers_diff, nouns_diff
    elif word_diff > WORD_COUNT_TOLERANCE:
        reason = f"word_count_mismatch: original_words:{len(orig_words)} vs correction_words:{len(corr_words)}"
        return original_text, "reject", reason, base_sim, sim_val, len_ratio, nli_label, nli_prob, numbers_diff, nouns_diff

    if base_sim is not None and base_sim < current_base_threshold:
        reason = f"base_similarity_too_low ({base_sim:.3f} < {current_base_threshold})"
        return original_text, "reject", reason, base_sim, sim_val, len_ratio, nli_label, nli_prob, numbers_diff, nouns_diff

    if len_ratio > LENGTH_CHANGE_ALLOWED_RATIO:
        reason = f"length_changed_too_much ({len_ratio:.3f} > {LENGTH_CHANGE_ALLOWED_RATIO})"
        return original_text, "reject", reason, base_sim, sim_val, len_ratio, nli_label, nli_prob, numbers_diff, nouns_diff

    if sim_val < ACCEPT_SIMILARITY_THRESHOLD:
        reason = f"low_similarity ({sim_val:.3f} < {ACCEPT_SIMILARITY_THRESHOLD})"
        return original_text, "reject", reason, base_sim, sim_val, len_ratio, nli_label, nli_prob, numbers_diff, nouns_diff

    if original_text.endswith('?') and not proposal.endswith('?'):
        reason = "question_mark_removed"
        return original_text, "reject", reason, base_sim, sim_val, len_ratio, nli_label, nli_prob, numbers_diff, nouns_diff

    original_numbers = extract_numbers(original_text)
    corrected_numbers = extract_numbers(proposal)
    nums_diff = original_numbers.symmetric_difference(corrected_numbers)
    if nums_diff:
        numbers_diff = ";".join(sorted(nums_diff))
        reason = f"numbers_altered: {numbers_diff}"
        return original_text, "reject", reason, base_sim, sim_val, len_ratio, nli_label, nli_prob, numbers_diff, nouns_diff

    original_nouns = extract_proper_nouns_ner(original_text)
    corrected_nouns = extract_proper_nouns_ner(proposal)
    nouns_diff_set = original_nouns.symmetric_difference(corrected_nouns)
    if nouns_diff_set:
        nouns_diff = ";".join(sorted(nouns_diff_set))
        reason = f"proper_nouns_altered: {nouns_diff}"
        return original_text, "reject", reason, base_sim, sim_val, len_ratio, nli_label, nli_prob, numbers_diff, nouns_diff

    try:
        nli_label, nli_prob = verify_with_nli_label_prob(original_text, proposal)
    except Exception as e:
        reason = f"nli_error: {e}"
        return original_text, "reject", reason, base_sim, sim_val, len_ratio, None, 0.0, numbers_diff, nouns_diff

    if nli_label is None:
        reason = "nli_no_label"
        return original_text, "reject", reason, base_sim, sim_val, len_ratio, nli_label, nli_prob, numbers_diff, nouns_diff

    # Nới lỏng NLI requirement - chấp nhận cả "neutral" với probability cao
    if nli_label.lower() not in ["entailment", "neutral"]:
        reason = f"nli_{nli_label} (prob={nli_prob:.3f})"
        return original_text, "reject", reason, base_sim, sim_val, len_ratio, nli_label, nli_prob, numbers_diff, nouns_diff
    elif nli_label.lower() == "neutral" and nli_prob < 0.7:
        # Nếu là neutral nhưng confidence thấp thì từ chối
        reason = f"nli_neutral_low_confidence (prob={nli_prob:.3f})"
        return original_text, "reject", reason, base_sim, sim_val, len_ratio, nli_label, nli_prob, numbers_diff, nouns_diff

    reason = f"accept (sim={sim_val:.3f}, base_sim={base_sim:.3f}, len_ratio={len_ratio:.3f}, nli_prob={nli_prob:.3f})"
    return proposal, "accept", reason, base_sim, sim_val, len_ratio, nli_label, nli_prob, numbers_diff, nouns_diff

# ---------------- Debug helpers ----------------

def ensure_debug_columns(df):
    cols = ["id","type","output","proposal","decision","reason","base_sim","sim","len_ratio","nli_label","nli_prob","numbers_changed","nouns_changed","orig_token_len","prop_token_len","truncated_reason"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df

def log_debug(debug_df, id_value, col_type, original, proposal, decision, reason,
              base_sim=None, sim=None, len_ratio=None, nli_label=None, nli_prob=None, numbers_changed=None, nouns_changed=None,
              orig_token_len=None, prop_token_len=None, truncated_reason=None):
    row = {
        "id": id_value,
        "type": col_type,
        "output": original,
        "proposal": proposal,
        "decision": decision,
        "reason": reason,
        "base_sim": base_sim,
        "sim": sim,
        "len_ratio": len_ratio,
        "nli_label": nli_label,
        "nli_prob": nli_prob,
        "numbers_changed": numbers_changed,
        "nouns_changed": nouns_changed,
        "orig_token_len": orig_token_len,
        "prop_token_len": prop_token_len,
        "truncated_reason": truncated_reason
    }
    # tạo DataFrame cho hàng mới và concat để tránh FutureWarning của pandas
    new_row_df = pd.DataFrame([row])
    debug_df = pd.concat([debug_df, new_row_df], ignore_index=True)
    # lưu ngay để resume / persist
    debug_df.to_csv(DEBUG_LOG, index=False, encoding="utf-8-sig")
    return debug_df


# ---------------- Save-on-exit ----------------

def save_on_exit(output_df, debug_df, created_internal_id=False, internal_id_col=None, sig=None, frame=None):
    print(f"\n[Ctrl+C] Stopped. Saving output to {OUTPUT_CSV} and debug to {DEBUG_LOG} ...")
    try:
        to_save = output_df.copy()
        if created_internal_id and internal_id_col in to_save.columns:
            to_save = to_save.drop(columns=[internal_id_col])
        to_save.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        debug_df.to_csv(DEBUG_LOG, index=False, encoding="utf-8-sig")
        print("Saved successfully.")
    except Exception as e:
        print("Save failed:", e)
    sys.exit(0)

# ---------------- MAIN ----------------

def main(args):
    # check input exists
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"INPUT CSV not found: {args.input_csv}")
    input_df = pd.read_csv(args.input_csv)
    n_rows = len(input_df)

    # determine id column (prefer 'id', then 'ID')
    if 'id' in input_df.columns:
        id_col = 'id'
        created_internal_id = False
    elif 'ID' in input_df.columns:
        id_col = 'ID'
        created_internal_id = False
    else:
        id_col = '_row_id'
        input_df[id_col] = list(range(n_rows))
        created_internal_id = True
        print("Warning: no 'id' column in INPUT_CSV. Created internal '_row_id' (will not be saved to OUTPUT).")

    # detect label column
    if 'label' in input_df.columns:
        label_col = 'label'
        label_present = True
    elif 'labels' in input_df.columns:
        label_col = 'labels'
        label_present = True
    else:
        label_col = 'predict_label'
        input_df[label_col] = ""
        label_present = False
        print("Info: no 'label' column in INPUT_CSV. Created 'predict_label' column (empty).")

    # prepare output_df (resume if exists)
    if os.path.exists(args.output_csv):
        output_df = pd.read_csv(args.output_csv)
        if len(output_df) != n_rows:
            print("Warning: OUTPUT row count != INPUT row count. Reinitializing output_df from input.")
            output_df = input_df.copy()
    else:
        output_df = input_df.copy()

    # debug log
    if os.path.exists(args.debug_log):
        debug_df = pd.read_csv(args.debug_log)
        debug_df = ensure_debug_columns(debug_df)
    else:
        debug_df = pd.DataFrame(columns=["id","type","output","proposal","decision","reason","base_sim","sim","len_ratio","nli_label","nli_prob","numbers_changed","nouns_changed","orig_token_len","prop_token_len","truncated_reason"])

    # build processed_set from debug_df (use string id for robustness)
    processed_set = set()
    if not debug_df.empty:
        for _, r in debug_df.iterrows():
            processed_set.add((str(r["id"]), r["type"]))

    # register Ctrl+C handler
    signal.signal(signal.SIGINT, lambda sig, frame: save_on_exit(output_df, debug_df, created_internal_id, id_col, sig, frame))

    print("Start processing rows...")

    # main: process rows in batches; for each row process context/prompt/response together
    for start in tqdm(range(0, n_rows, ROWS_BATCH_SIZE), desc="Processing rows"):
        batch_rows = list(range(start, min(start + ROWS_BATCH_SIZE, n_rows)))

        prompts = []
        mapping = []  # list of (row_idx, idv, col, original_text, orig_token_len, truncated_reason)

        # Build prompts for all (row, col) in this batch that are not processed
        for row_idx in batch_rows:
            idv = input_df.at[row_idx, id_col]
            for col in COLUMNS_TO_FIX:
                if col not in input_df.columns:
                    continue
                key = (str(idv), col)
                if key in processed_set:
                    continue
                original_text = "" if pd.isna(input_df.at[row_idx, col]) else str(input_df.at[row_idx, col]).strip()

                # decide allowed tokens per column based on EDA
                if col == 'context':
                    allowed_tokens_for_text = CONTEXT_ALLOWED_TOKENS
                else:
                    allowed_tokens_for_text = SHORT_COL_ALLOWED_TOKENS

                orig_tlen = tokens_len(original_text, gen_tokenizer)
                truncated_reason = ""
                use_text = original_text
                if orig_tlen > allowed_tokens_for_text:
                    truncated_text, was_trunc, orig_len = truncate_text_by_tokens(original_text, allowed_tokens_for_text, gen_tokenizer, keep_head_ratio=0.6)
                    truncated_reason = f"truncated (orig_tokens={orig_len} > allowed={allowed_tokens_for_text})"
                    use_text = truncated_text

                prompt = make_correction_prompt(use_text)
                prompts.append(prompt)
                mapping.append((row_idx, idv, col, original_text, orig_tlen, truncated_reason))

        if not prompts:
            continue

        # generate proposals
        proposals = generate_from_model(prompts)

        # iterate mapping + proposals
        for (row_idx, idv, col, original_text, orig_tlen, truncated_reason), proposal in zip(mapping, proposals):
            # Clean model raw output: loại bỏ tiền tố instruction nếu model echo prompt
            proposal = _clean_model_output(proposal)
            
            prop_tlen = tokens_len(proposal, gen_tokenizer)
            final_text, decision, reason, base_sim, sim_val, len_ratio, nli_label, nli_prob, numbers_diff, nouns_diff = validate_proposal_with_nli(original_text, proposal, col)

            # write final_text into output_df (keep original if rejected)
            output_df.at[row_idx, col] = final_text

            # mark processed and log debug
            processed_set.add((str(idv), col))
            debug_df = log_debug(debug_df, str(idv), col, original_text, proposal, decision, reason,
                     base_sim=base_sim, sim=sim_val, len_ratio=len_ratio,
                     nli_label=nli_label, nli_prob=nli_prob,
                     numbers_changed=numbers_diff, nouns_changed=nouns_diff,
                     orig_token_len=orig_tlen, prop_token_len=prop_tlen, truncated_reason=truncated_reason)


        # save after batch; drop internal id if created before saving
        to_save = output_df.copy()
        if created_internal_id and id_col in to_save.columns:
            to_save = to_save.drop(columns=[id_col])
        to_save.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
        debug_df.to_csv(args.debug_log, index=False, encoding="utf-8-sig")

    # final save
    final_save = output_df.copy()
    if created_internal_id and id_col in final_save.columns:
        final_save = final_save.drop(columns=[id_col])
    final_save.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    debug_df.to_csv(args.debug_log, index=False, encoding="utf-8-sig")

    print("\nAll done.")
    print("Output saved to:", args.output_csv)
    print("Debug log saved to:", args.debug_log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix spelling with NLI validation (per-column)")
    parser.add_argument("--input_csv", default=INPUT_CSV)
    parser.add_argument("--output_csv", default=OUTPUT_CSV)
    parser.add_argument("--debug_log", default=DEBUG_LOG)
    parser.add_argument("--dry_run", action="store_true", help="If set, only build prompts and show stats without calling models")
    args = parser.parse_args()

    if args.dry_run:
        # Dry-run: show a few prompt previews and token stats (no model calls)
        if not os.path.exists(args.input_csv):
            raise FileNotFoundError(f"INPUT CSV not found: {args.input_csv}")
        df = pd.read_csv(args.input_csv)
        n = min(10, len(df))
        print("Dry-run token stats and sample prompts (first", n, "rows):")
        for i in range(n):
            row = df.iloc[i]
            print("--- Row", i, "id=", row.get('id', row.get('ID', i)))
            for col in COLUMNS_TO_FIX:
                text = "" if pd.isna(row.get(col, "")) else str(row.get(col))
                tlen = tokens_len(text, gen_tokenizer)
                allowed = CONTEXT_ALLOWED_TOKENS if col == 'context' else SHORT_COL_ALLOWED_TOKENS
                print(f"  {col}: tokens={tlen}, allowed={allowed}")
                if tlen > allowed:
                    truncated_text, _, orig_len = truncate_text_by_tokens(text, allowed, gen_tokenizer)
                    print("    -> would truncate (preview):", truncated_text[:200])
        sys.exit(0)

    main(args)