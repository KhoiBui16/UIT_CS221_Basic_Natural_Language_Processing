# Project Architecture Overview

```mermaid
flowchart TD
    subgraph Init[1. Khởi tạo]
        A1[Định nghĩa Config\n- Đường dẫn\n- MODEL_NAME\n- Hyperparams (lr, batch, epochs...)]
        A2[Thiết lập Logger\n- Console/File handlers\n- Ghi sự kiện]
        A1 --> A2
    end

    subgraph DataPrep[2. Chuẩn bị Dữ liệu · prepare_data]
        B1[Đọc vihallu-train.csv bằng Pandas]
        B2[Tiền xử lý chuỗi:\nprompt </s></s> response </s></s> context]
        B3[Ánh xạ nhãn:\nintrinsic→0, extrinsic→1, no→2]
        B4[Chia train/val 80/20 (stratify)]
        B5[Lưu train_split.csv & validation_split.csv]
        B1 --> B2 --> B3 --> B4 --> B5
    end

    subgraph ModelLoad[3. Model & Tokenizer]
        C1[Tải Tokenizer\nAutoTokenizer.from_pretrained]
        C2[Tải Model\nAutoModelForSequenceClassification]
        C3[Điều chỉnh Dropout theo Config]
        C1 --> C2 --> C3
    end

    subgraph DatasetStage[4. Dataset & DataLoader]
        D1[Tạo HallucinationDataset]
        D2[__getitem__ → tokenize input_text,\ntrunc/pad tới MAX_LENGTH]
        D3[Tạo DataLoader train/val\n(batch size, shuffle)]
        D1 --> D2 --> D3
    end

    subgraph TrainingSetup[5. Thiết lập Huấn luyện]
        E1[Move model to CUDA]
        E2[Khởi tạo AdamW\n(lr, weight decay)]
        E3[Scheduler cosine\n+ warmup steps]
        E4[CrossEntropyLoss\nvới class weights]
        E1 --> E2 --> E3 --> E4
    end

    subgraph Loop[6. Vòng lặp Huấn luyện · EPOCHS]
        F1[train_one_epoch\n- Gradient accumulation\n- Clip grad\n- Cập nhật optimizer+scheduler]
        F2[evaluate trên val_loader]
        F3[Tính Macro-F1 & Accuracy]
        F4[Checkpoint nếu Macro-F1 cải thiện\nlưu model & tokenizer]
        F5[Early Stopping nếu macro-F1\nkhông cải thiện > patience_limit]
        F1 --> F2 --> F3 --> F4
        F3 --> F5
    end

    subgraph End[7. Kết thúc]
        G1[Log Macro-F1 tốt nhất]
        G2[Thông báo đường dẫn checkpoint]
        G1 --> G2
    end

    Init --> DataPrep --> ModelLoad --> DatasetStage --> TrainingSetup --> Loop --> End
```

## Architectural Stages
- **Stage 0 · Inputs & Governance**: Nguồn dữ liệu duy nhất từ ban tổ chức, chuẩn Metric Macro-F1/Accuracy, và phần cấu hình bảo mật trong `.env` tạo nền tảng tuân thủ quy định.
- **Stage 1 · Data Management & Curation**: Duy trì catalog dữ liệu gốc, module `prepare_data` chuẩn hóa text + phân chia stratified để giảm lệch lớp, đồng thời lưu lại các split phục vụ kiểm thử và reproducibility.
- **Stage 2 · Feature Construction**: Thiết kế template prompt–response–context, ánh xạ nhãn và trọng số để cân bằng lớp, đóng gói thành `HallucinationDataset` và DataLoader với metadata phục vụ gradient accumulation.
- **Stage 3 · Model Adaptation**: Tầng tải checkpoint XNLI, cấu hình model/tokenizer, lựa chọn lịch học (AdamW + cosine warmup), và điều khiển thí nghiệm qua `Config`, logger, seed đặt trong notebook.
- **Stage 4 · Evaluation & Selection**: Vòng huấn luyện và đánh giá sinh metrics chính, so sánh Macro-F1 để lưu checkpoint tốt nhất, quản lý early stopping nhằm tránh overfitting.
- **Stage 5 · Observability & Delivery**: Ghi log chi tiết, lưu trữ artefact (model, tokenizer, config) và chuẩn bị bước inference/submission tiếp theo; đây là nơi bạn sẽ mở rộng cho pipeline suy luận hoặc packaging.
