# app/app.py
import sys
import os
import streamlit as st
import pandas as pd
import time
import torch
import altair as alt

# Đảm bảo Python tìm thấy inference_module.py dù chạy ở đâu
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Nếu file này nằm ở root, cần thêm thư mục app/ vào path
app_dir = os.path.join(current_dir, "app")
if os.path.exists(app_dir) and app_dir not in sys.path:
    sys.path.append(app_dir)

try:
    # Thử import trực tiếp (nếu app.py nằm cùng thư mục với inference_module.py)
    from inference_module import run_inference, load_model_and_tokenizer
except ImportError:
    # Fallback (nếu app.py nằm ở root và inference_module nằm trong app/)
    from app.inference_module import run_inference, load_model_and_tokenizer

# --- CẤU HÌNH ---
st.set_page_config(
    page_title="Hallucination Detection Classification", page_icon="🕵️", layout="wide"
)


# --- CACHE MODEL ---
@st.cache_resource(show_spinner="⏳ Đang tải model (lần đầu sẽ mất vài phút)...")
def get_model(model_name, device_str):
    device = torch.device(device_str)
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    return model, tokenizer, device


st.title("🕵️ Vietnamese LLM Hallucination Classification")
st.caption("Chạy trực tiếp mô hình trên Streamlit không qua API.")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Cấu hình Model")
MODEL_OPTIONS = {
    "🏆 xlm-roberta-large(Recommended - 1)": "KhoiBui/xlm-roberta-large-hallucination-classification",
    "🤖 xlm-roberta-large-xnli (Recommended - 2)": "KhoiBui/xlm-roberta-large-xnli-hallucination-classification",
    "☕ CafeBERT (Pure Vietnamese)": "KhoiBui/CafeBERT-hallucination-classification",
    "🌐 infoxlm-large (Multilingual)": "KhoiBui/infoxlm-large-hallucination-classification",
}
selected_label = st.sidebar.selectbox("Chọn mô hình:", list(MODEL_OPTIONS.keys()))
selected_model_id = MODEL_OPTIONS[selected_label]

device_opt = st.sidebar.radio(
    "Thiết bị chạy:", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
st.sidebar.info(f"Model ID: `{selected_model_id}`\n\nDevice: `{device_opt.upper()}`")


# --- HÀM DỰ ĐOÁN TRỰC TIẾP ---
def run_prediction_direct(df, model_id, device_str):
    status_text = st.empty()
    progress_bar = st.progress(0, text="⏳ Đang chờ model...")

    start_time = time.time()
    try:
        # Dòng này sẽ tự động hiển thị spinner (từ @st.cache_resource) NẾU model chưa được tải
        model, tokenizer, device = get_model(model_id, device_str)

        # Ngay khi model sẵn sàng, cập nhật progress bar
        progress_bar.progress(30, text="🧠 Đang phân tích...")
        result_df = run_inference(model, tokenizer, df, device)

        progress_bar.progress(100, text="✅ Hoàn tất!")
        exec_time = time.time() - start_time
        st.toast(f"Hoàn tất trong {exec_time:.2f}s!")
        time.sleep(1)
        progress_bar.empty()

        return result_df

    except Exception as e:
        progress_bar.empty()
        st.error(f"Lỗi khi chạy model: {e}")
        return None


# --- GIAO DIỆN CHÍNH ---
tab1, tab2 = st.tabs(["📝 Kiểm tra nhanh", "📂 Upload file CSV"])

with tab1:
    col1, col2 = st.columns(2)
    context = col1.text_area("Context:", height=300, placeholder="Ngữ cảnh...")
    prompt = col2.text_area("Prompt:", height=300, placeholder="Câu hỏi...")
    response = st.text_area(
        "Response:", height=200, placeholder="Câu trả lời của LLM..."
    )

    if st.button("🔍 Kiểm tra ngay", type="primary", width="stretch"):
        if all([context.strip(), prompt.strip(), response.strip()]):
            df_in = pd.DataFrame(
                [{"context": context, "prompt": prompt, "response": response}]
            )
            res_df = run_prediction_direct(df_in, selected_model_id, device_opt)

            if res_df is not None:
                row = res_df.iloc[0]
                lbl, score = row.get("predict_label", "N/A"), row.get("score", 0.0)
                st.markdown("### Kết quả:")

                if lbl == "no":
                    st.success(
                        f"✅ **NO** (Không ảo giác, phản hồi hoàn toàn phù hợp và chỉ dựa vào ngữ cảnh.)  (Score: {score:.1%})"
                    )
                elif lbl == "intrinsic":
                    st.error(  # Đổi thành error (màu đỏ)
                        f"❌ **INTRINSIC** (Phản hồi mâu thuẫn hoặc bóp méo thông tin so với ngữ cảnh.)  (Score: {score:.1%})"
                    )
                else:
                    st.warning(  # Đổi thành warning (màu vàng)
                        f"⚠️ **EXTRINSIC** (Phản hồi bổ sung thông tin không có căn cứ hoặc không thể truy xuất từ ngữ cảnh.) (Score: {score:.1%})"
                    )
        else:
            st.warning("Vui lòng nhập đủ thông tin.")

with tab2:
    st.header("📂 Kiểm tra theo lô (Batch Processing)")
    uploaded = st.file_uploader(
        "Upload CSV (cần cột: context, prompt, response)", type=["csv"]
    )
    if uploaded:
        df = pd.read_csv(uploaded)
        df.rename(columns=lambda x: x.strip().lower(), inplace=True)
        if {"context", "prompt", "response"}.issubset(df.columns):

            if "num_samples" not in st.session_state:
                st.session_state.num_samples = min(10, len(df))

            st.session_state.num_samples = st.number_input(
                f"Nhập số lượng mẫu muốn test (tối đa: {len(df)} mẫu)",
                min_value=1,
                max_value=len(df),
                step=10,
                value=st.session_state.num_samples,
                key="num_samples_input",
            )

            num_samples = st.session_state.num_samples
            df_test = df.head(num_samples).copy()

            if st.button(
                f"🚀 Chạy {num_samples} mẫu đầu tiên",
                type="primary",
                key="run_batch",
                width="stretch",
            ):
                res_df = run_prediction_direct(df_test, selected_model_id, device_opt)
                if res_df is not None:
                    if "predict_label" in res_df.columns:

                        st.subheader("📊 Thống kê kết quả")
                        c1, c2 = st.columns([1, 2])

                        with c1:
                            counts = res_df["predict_label"].value_counts()
                            total = len(res_df)
                            st.markdown("#### Phân bố nhãn")
                            for label, count in counts.items():
                                st.metric(
                                    str(label).upper(),
                                    f"{count}",
                                    f"{(count / total * 100):.1f}%",
                                )

                        with c2:
                            chart_data = counts.reset_index()
                            chart_data.columns = ["Label", "Count"]

                            domain = ["no", "extrinsic", "intrinsic"]
                            range_ = [
                                "#28a745",
                                "#ffc107",
                                "#dc3545",
                            ]  # Green, Yellow, Red
                            color_scale = alt.Scale(domain=domain, range=range_)

                            chart = (
                                alt.Chart(chart_data)
                                .mark_bar()
                                .encode(
                                    x=alt.X("Label", axis=None),
                                    y=alt.Y("Count", title="Số lượng mẫu"),
                                    color=alt.Color(
                                        "Label",
                                        scale=color_scale,
                                        legend=alt.Legend(title="Loại nhãn"),
                                    ),
                                    tooltip=["Label", "Count"],
                                )
                                .properties(title="Biểu đồ phân bố kết quả")
                                .interactive()
                            )

                            st.altair_chart(chart, width="stretch")

                    st.subheader("📋 Chi tiết kết quả")
                    st.dataframe(res_df, width="stretch")
        else:
            st.error(
                f"File CSV thiếu các cột bắt buộc. Cần có: `context`, `prompt`, `response`."
            )

# streamlit run app/app.py
