import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import time

# Load biến môi trường từ file .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env. Please check again!")

# Khởi tạo model (SDK mới chỉ cần configure 1 lần)
genai.configure(api_key=GEMINI_API_KEY)
# for m in genai.list_models():
#     print(m.name, " -> ", m.supported_generation_methods)

# Sử dụng model flash để tăng rate limit và tốc độ
model = genai.GenerativeModel("gemini-2.5-flash")
print("Gemini API configured successfully!")

corrected_ids = set()

def correct_vietnamese_spelling(text, row_id, col_name):
    if pd.isna(text) or str(text).strip() == "":
        return text

    prompt = f"""Bạn là một chuyên gia ngữ pháp tiếng Việt.
Nhiệm vụ của bạn là kiểm tra và sửa tất cả các lỗi chính tả, lỗi ngữ pháp, và lỗi dùng từ trong đoạn văn bản tiếng Việt sau.
Chỉ trả về đoạn văn bản đã được sửa lỗi hoàn chỉnh và chính xác, không thêm bất kỳ lời mở đầu, kết thúc, giải thích hay thông tin bổ sung nào khác.

Văn bản gốc cần sửa:
{text}

Văn bản đã được sửa lỗi chính tả và ngữ pháp:"""
    retries = 3
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            corrected_text = response.text.strip() if response.text else text

            if corrected_text != text:
                corrected_ids.add(row_id)
                print(f"    -> Đã sửa chính tả cho cột '{col_name}' (ID: {row_id}).")

            return corrected_text
        except Exception as e:
            if "429" in str(e):
                wait_time = 30
                print(f"Lỗi 429 (quota exceeded), đang chờ {wait_time}s trước khi thử lại...")
                time.sleep(wait_time)
            else:
                print(f"Lỗi khi sửa chính tả cho ID {row_id}, cột {col_name}: {e}")
                return text
    return text

def evaluation_hallucination(context, prompt, response, row_id):
    context = str(context) if not pd.isna(context) else ""
    prompt = str(prompt) if not pd.isna(prompt) else ""
    response = str(response) if not pd.isna(response) else ""

    evaluation_prompt = f"""Bạn là một chuyên gia phân tích chất lượng thông tin và tính chính xác của phản hồi từ mô hình ngôn ngữ.
Dưới đây là một Context (Ngữ cảnh), một Prompt (Yêu cầu) và một Response (Phản hồi).
Nhiệm vụ của bạn là đánh giá xem Response có chứa thông tin bịa đặt (hallucination) so với thông tin đã được cung cấp trong Context và Prompt hay không.

Hãy sử dụng các định nghĩa sau để đưa ra nhãn đánh giá:
- 'no': Response hoàn toàn chính xác, nhất quán và chỉ chứa thông tin có thể suy ra từ Context và Prompt.
- 'extrinsic': Response chứa thông tin không có trong Context hoặc Prompt, nhưng thông tin bổ sung này là đúng sự thật và không mâu thuẫn trực tiếp.
- 'intrinsic': Response chứa thông tin mâu thuẫn trực tiếp hoặc hoàn toàn sai lệch.

Chỉ trả về duy nhất một trong ba nhãn sau: 'no', 'extrinsic', 'intrinsic'.

---
# Context (Ngữ cảnh):
{context}

# Prompt (Yêu cầu):
{prompt}

# Response (Phản hồi):
{response}

Nhãn đánh giá:"""

    retries = 3
    for attempt in range(retries):
        try:
            gemini_response = model.generate_content(evaluation_prompt)
            label = gemini_response.text.strip().lower()

            if label in ['no', 'extrinsic', 'intrinsic']:
                return label
            else:
                print(f"  Cảnh báo: Gemini trả về nhãn không hợp lệ '{label}' cho ID {row_id}.")
                return "no_label_error"
        except Exception as e:
            if "429" in str(e):
                wait_time = 30
                print(f"Lỗi 429 (quota exceeded), đang chờ {wait_time}s trước khi thử lại...")
                time.sleep(wait_time)
            else:
                print(f"Lỗi khi đánh giá vi phạm cho ID {row_id}: {e}")
                return "error_evaluation"
    return "error_evaluation"


# 3. XỬ LÝ FILE CSV CHÍNH

INPUT_CSV_PATH = 'vihallu-public-test.csv'
OUTPUT_CSV_PATH = 'fixed-vihallu-public-test.csv'

try:
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"Đã đọc thành công file: {INPUT_CSV_PATH}. Tổng số mẫu: {len(df)}")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{INPUT_CSV_PATH}'. Vui lòng đảm bảo file nằm trong cùng thư mục với script.")
    exit()
except Exception as e:
    print(f"Lỗi khi đọc file CSV: {e}")
    exit()

# Ép kiểu predict_label sang str để tránh FutureWarning
if 'predict_label' in df.columns:
    df['predict_label'] = df['predict_label'].astype(str)
else:
    df['predict_label'] = ""

columns_to_correct = ['context', 'prompt', 'response']

for index, row in df.iterrows():
    current_id = row['id']
    print(f"\n--- Đang xử lý hàng {index + 1}/{len(df)} (ID: {current_id}) ---")

    # Nhiệm vụ 1: Sửa lỗi chính tả
    for col in columns_to_correct:
        original_text = row[col]
        if pd.isna(original_text):
            continue
        corrected_text = correct_vietnamese_spelling(original_text, current_id, col)
        df.at[index, col] = corrected_text
        time.sleep(2)  # tăng delay tránh 429

    # Nhiệm vụ 2: Đánh giá hallucination
    context_text = df.at[index, 'context']
    prompt_text = df.at[index, 'prompt']
    response_text = df.at[index, 'response']

    predicted_label = evaluation_hallucination(context_text, prompt_text, response_text, current_id)
    df.at[index, 'predict_label'] = predicted_label
    print(f"  Nhãn dự đoán cho predict_label (ID: {current_id}): {predicted_label}")
    time.sleep(2)  # tăng delay tránh 429

# 4. LƯU KẾT QUẢ

try:
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"\nQuá trình xử lý hoàn tất! Kết quả đã được lưu vào: {OUTPUT_CSV_PATH}")
except Exception as e:
    print(f"Lỗi khi lưu file CSV: {e}")

if corrected_ids:
    print(f"\n--- Danh sách các ID có ít nhất một lỗi chính tả đã được sửa ---")
    for _id in sorted(list(corrected_ids)):
        print(f"- {_id}")
else:
    print(f"\nKhông có ID nào được ghi nhận là đã sửa lỗi chính tả trong quá trình xử lý.")
