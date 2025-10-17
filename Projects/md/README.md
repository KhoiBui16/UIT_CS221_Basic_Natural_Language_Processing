Tổng quan Pipeline và Phương pháp Phát hiện Hallucination
==========================================================

Chào cả nhóm, đây là tóm tắt pipeline và các kỹ thuật mà đội mình đã xây dựng để giải quyết bài toán này. Hướng tiếp cận chính là Ensemble có trọng số (Weighted Ensemble) dựa trên các mô hình Transformer mạnh mẽ thuộc nhiều kiến trúc khác nhau.

1. Tiền xử lý và Chuẩn bị Dữ liệu
-----------------------------------

*   **Phương pháp:** Ghép nối có định dạng (Formatted Concatenation).
*   **Giải thích:** Thay vì đưa từng phần (`context`, `prompt`, `response`) vào riêng lẻ, chúng ta ghép chúng thành một chuỗi duy nhất theo một khuôn mẫu rõ ràng: `Ngữ cảnh: [context] [SEP] Câu hỏi: [prompt] [SEP] Trả lời: [response]`.
    *   **Lợi ích:** Cách làm này cho phép các mô hình dựa trên Transformer (như RoBERTa, DeBERTa) sử dụng cơ chế Cross-Attention để so sánh chéo thông tin giữa các phần, giúp phát hiện mâu thuẫn (`intrinsic`) và thông tin bịa đặt (`extrinsic`) một cách hiệu quả.

2. Huấn luyện các Mô hình Nền tảng (Foundation Models)
----------------------------------------------------------

Chúng ta đã huấn luyện nhiều mô hình độc lập để tìm ra những "chuyên gia" tốt nhất, mỗi mô hình có một thế mạnh riêng.

*   **Phương pháp:** Supervised Fine-tuning (Tinh chỉnh có giám sát).
*   **Giải thích:** Chúng ta sử dụng các mô hình ngôn ngữ đã được huấn luyện trước (pre-trained) và tiếp tục huấn luyện (fine-tune) chúng trên bộ dữ liệu 7000 mẫu của cuộc thi cho tác vụ phân loại 3 lớp (`no`, `intrinsic`, `extrinsic`).

    **a) Fine-tuning các mô hình Encoder-only:**
    *   **Mô hình sử dụng:**
        1.  **`microsoft/infoxlm-large`:** Mạnh nhất trên tập validation (Val F1 ≈ 0.805). Đây là chuyên gia bắt lỗi `no` và `intrinsic`.
        2.  **`joeddav/xlm-roberta-large-xnli`:** Rất mạnh và ổn định (Val F1 ≈ 0.777). Mô hình này đã được pre-train cho tác vụ suy luận (NLI), nên nó là chuyên gia bắt lỗi `extrinsic`.
        3.  **`FacebookAI/xlm-roberta-large`:** Một mô hình "đa năng", hoạt động tốt và ổn định (Val F1 ≈ 0.782).
    *   **Kỹ thuật Tối ưu:**
        *   **Learning Rate Thấp & Warmup:** Sử dụng learning rate rất nhỏ (`1e-5` đến `5e-6`) và có giai đoạn warmup để giúp các mô hình lớn hội tụ ổn định.
        *   **Gradient Clipping:** Ngăn chặn hiện tượng "bùng nổ gradient", giúp quá trình học không bị chệch hướng.
        *   **Huấn luyện Dài hơn:** Tăng số epochs lên 10 để tìm ra điểm hiệu năng tốt nhất trước khi overfitting.

    **b) Fine-tuning Mô hình Ngôn ngữ Lớn (LLM):**
    *   **Mô hình sử dụng:** **`Viet-Mistral/Vistral-7B-Chat`**.
    *   **Phương pháp:** **QLoRA (Quantized Low-Rank Adaptation)**.
    *   **Giải thích:**
        1.  **Lượng tử hóa (Quantization):** Toàn bộ mô hình Vistral-7B được nén xuống và load ở dạng 4-bit để tiết kiệm VRAM.
        2.  **Adapter LoRA:** Thay vì huấn luyện toàn bộ 7 tỷ tham số, chúng ta chỉ "đóng băng" mô hình gốc và huấn luyện các "adapter" (bộ điều hợp) nhỏ được thêm vào các lớp attention.
        3.  **Instruction Fine-tuning:** Chúng ta định dạng lại bài toán thành một cuộc hội thoại. Mỗi mẫu dữ liệu được chuyển thành một prompt chỉ dẫn rõ ràng, và mô hình được dạy để sinh ra câu trả lời là một trong ba nhãn.
    *   **Lợi ích:** Kỹ thuật này cho phép chúng ta tận dụng sức mạnh suy luận của LLM mà không cần đến dàn máy chủ GPU đắt tiền.

3. Kỹ thuật Ensemble và Xử lý Hậu kỳ
-------------------------------------------

Đây là bước quyết định để tối đa hóa điểm số.

*   **Phương pháp:** **Weighted Soft Voting** + **Heuristic-based Post-processing**.
*   **Giải thích:**
    1.  **Soft Voting:** Thay vì chỉ lấy nhãn dự đoán, chúng ta lấy ra **vector xác suất** (probabilities) từ mỗi mô hình cho mỗi mẫu dữ liệu.
    2.  **Weighted Averaging:** Các vector xác suất này được nhân với một **trọng số** (tùy thuộc vào độ mạnh của từng mô hình, được tối ưu trên tập validation) rồi cộng lại với nhau.
    3.  **Dự đoán cuối cùng:** Nhãn có xác suất tổng hợp cao nhất sẽ được chọn.
        *   *Ví dụ:* `Final_Prob = 0.4*Prob_InfoXLM + 0.4*Prob_XNLI + 0.2*Prob_Vistral`
    4.  **Post-processing (Luật "Số Mới"):** Sau khi có kết quả từ ensemble, chúng ta áp dụng một quy tắc cuối cùng: Nếu dự đoán là `no` nhưng `response` lại chứa một con số không có trong `context`, chúng ta sẽ sửa lại dự đoán đó thành `extrinsic`.

**Kết quả:** Sự kết hợp giữa các mô hình có kiến trúc và "sở trường" khác nhau (Encoder, LLM), cùng với việc tối ưu trọng số và áp dụng luật kinh nghiệm, đã giúp tạo ra một hệ thống dự đoán mạnh mẽ, ổn định và có khả năng tổng quát hóa tốt.