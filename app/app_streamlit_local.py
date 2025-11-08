# app/app_streamlit_local.py
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import streamlit as st
import pandas as pd
import requests
import json
import time
import altair as alt

# --- CẤU HÌNH CỔNG MỚI (8095) ---
API_URL = "http://localhost:8095/predict_batch"
HEALTH_URL = "http://localhost:8095/health"
FIXED_MODEL_ID = None

st.set_page_config(
    page_title="Hallucination Detection Hub", page_icon="🕵️", layout="wide"
)

st.title("🕵️ Vietnamese LLM Hallucination Detection")
st.caption("Công cụ kiểm tra độ trung thực của câu trả lời từ mô hình ngôn ngữ lớn.")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Cấu hình")

if FIXED_MODEL_ID:
    selected_model_id = FIXED_MODEL_ID
    st.sidebar.info(f"🔒 Model đang sử dụng:\n\n`{selected_model_id}`")
else:
    MODEL_OPTIONS = {
        "🏆 XLM-R Large (Recommended)": "KhoiBui/xlm-roberta-large-hallucination-classification",
        "☕ CafeBERT (Pure Vietnamese)": "KhoiBui/CafeBERT-hallucination-classification",
        "🌐 InfoXLM Large (Multilingual)": "KhoiBui/infoxlm-large-hallucination-classification",
        "🤖 XLM-R Large XNLI": "KhoiBui/xlm-roberta-large-xnli-hallucination-classification",
    }
    selected_model_label = st.sidebar.selectbox(
        "Chọn mô hình:", list(MODEL_OPTIONS.keys())
    )
    selected_model_id = MODEL_OPTIONS[selected_model_label]
    st.sidebar.info(f"Model ID: `{selected_model_id}`")

with st.sidebar:
    st.markdown("---")
    try:
        # Timeout ngắn để không làm treo UI nếu backend chưa lên
        resp = requests.get(HEALTH_URL, timeout=1)
        if resp.status_code == 200:
            st.success(f"✅ API Online ({resp.json().get('device', 'unknown')})")
        else:
            st.warning("⚠️ API Unstable")
    except:
        st.error("❌ API Offline. Hãy chạy `python app/run_app.py`")


# --- HÀM GỌI API ---
def call_api(df: pd.DataFrame, model_name: str):
    # Chuẩn bị payload, đảm bảo không có giá trị null/NaN gây lỗi JSON
    df_clean = df.fillna("")
    data_records = df_clean[["context", "prompt", "response"]].to_dict(orient="records")

    payload = {
        "model_name": model_name,
        "data": data_records,
    }

    start_time = time.time()
    try:
        with st.spinner("🔄 Đang phân tích..."):
            response = requests.post(API_URL, json=payload, timeout=600)

        if response.status_code == 200:
            exec_time = time.time() - start_time
            st.toast(f"Hoàn thành trong {exec_time:.2f}s!", icon="🎉")
            return pd.DataFrame(response.json())
        else:
            st.error(f"Lỗi API ({response.status_code}): {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Không thể kết nối tới Backend API tại port 8095.")
        return None
    except Exception as e:
        st.error(f"Lỗi: {e}")
        return None


# --- GIAO DIỆN CHÍNH ---
tab1, tab2 = st.tabs(["📝 Kiểm tra nhanh", "📂 Upload file CSV"])

# TAB 1: SINGLE CHECK
with tab1:
    col1, col2 = st.columns(2)
    prompt = col1.text_area("Prompt:", height=100, placeholder="Câu hỏi...")
    context = col2.text_area(
        "Context:", height=100, placeholder="Ngữ cảnh minh chứng..."
    )
    response = st.text_area(
        "Response (cần kiểm tra):", height=100, placeholder="Câu trả lời của LLM..."
    )

    # FIX: Thay width="stretch" bằng width='stretch'
    if st.button("🔍 Kiểm tra", type="primary", width='stretch'):
        if all([context.strip(), prompt.strip(), response.strip()]):
            df_in = pd.DataFrame(
                [{"context": context, "prompt": prompt, "response": response}]
            )
            res_df = call_api(df_in, selected_model_id)

            if res_df is not None:
                row = res_df.iloc[0]
                lbl = row.get("predict_label", "N/A")
                score = row.get("score", 0.0)

                st.markdown("### Kết quả:")
                if lbl == "no":
                    st.success(f"✅ **HỢP LÝ** (Score: {score:.2f})")
                elif lbl == "intrinsic":
                    st.error(f"❌ **MÂU THUẪN** (Score: {score:.2f})")
                elif lbl == "extrinsic":
                    st.warning(f"⚠️ **BỊA ĐẶT** (Score: {score:.2f})")
                else:
                    st.info(f"Label: {lbl} (Score: {score:.2f})")
        else:
            st.warning("Vui lòng điền đầy đủ thông tin.")

# TAB 2: BATCH CSV
with tab2:
    st.header("📂 Kiểm tra theo lô (Batch Processing)")
    uploaded_file = st.file_uploader(
        "Upload CSV (cần cột: context, prompt, response)", type=["csv"]
    )
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            # Chuẩn hóa tên cột
            df.rename(columns=lambda x: x.strip().lower(), inplace=True)

            required = {"context", "prompt", "response"}
            if not required.issubset(df.columns):
                st.error(
                    f"File thiếu cột bắt buộc. Cần có: {required}. Hiện có: {set(df.columns)}"
                )
            else:
                
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
                
                placeholder = st.empty()

                num_samples = st.session_state.num_samples
                df_test = df.head(num_samples).copy()
                
                with placeholder.container():
                    with st.expander("👀 Xem trước dữ liệu", expanded=True):
                        st.dataframe(df_test, width='stretch')
                        st.caption(f"Tổng: {len(df)} dòng")

                if st.button(
                    f"🚀 Chạy {num_samples} mẫu đầu tiên",
                    type="primary",
                    key="run_batch",
                    width="stretch",
                ):
                    result_df = call_api(df_test, selected_model_id)
                    if result_df is not None:
                        st.success("✅ Hoàn tất!")
                        if "predict_label" in result_df.columns:
                            st.subheader("📊 Thống kê")
                            c1, c2 = st.columns([1, 2])
                            with c1:
                                counts = result_df["predict_label"].value_counts()
                                for label, count in counts.items():
                                    st.metric(
                                        str(label).upper(),
                                        f"{count}",
                                        f"{(count/len(result_df)*100):.1f}%",
                                    )
                            with c2:
                                # Dùng Altair cho đẹp
                                chart_data = counts.reset_index()
                                chart_data.columns = ["Label", "Count"]
                                domain = ["no", "extrinsic", "intrinsic"]
                                range_ = ["#28a745", "#ffc107", "#dc3545"]
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

                                st.altair_chart(
                                    chart, width='stretch'
                                )  # FIX: width="stretch" -> width='stretch'

                        st.subheader("📋 Chi tiết")
                        st.dataframe(
                            result_df,
                            # FIX: Thay width="stretch" bằng width='stretch'
                            width='stretch',
                            column_config={
                                "score": st.column_config.ProgressColumn(
                                    "Confidence",
                                    format="%.4f",
                                    min_value=0,
                                    max_value=1,
                                )
                            },
                        )
                        st.download_button(
                            "📥 Tải kết quả CSV",
                            result_df.to_csv(index=False).encode("utf-8-sig"),
                            f"results_{int(time.time())}.csv",
                            "text/csv",
                            type="primary",
                            # FIX: Thay width="stretch" bằng width='stretch'
                            width='stretch',
                        )
        except Exception as e:
            st.error(f"Lỗi đọc file CSV: {e}")
