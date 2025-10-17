# config_temp_base.py
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Đường dẫn và Tên file ---
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_FILE = os.path.join(DATA_DIR, "vihallu-train.csv") 
TEST_FILE =  os.path.join(DATA_DIR,  "vihallu-public-test.csv") 

SUBMISSION_DIR = os.path.join(ROOT_DIR, "submission") 
SUBMISSION_CSV = "submit.csv"
SUBMISSION_ZIP = "submit.zip"

# --- Cấu hình Mô hình ---
# MODEL_NAME = "joeddav/xlm-roberta-large-xnli"   # (0.7851 - F1)
# MODEL_NAME = "FacebookAI/xlm-roberta-large" # (0. 78192 - F1)
# MODEL_NAME = "microsoft/infoxlm-large" (0.7740 - F1)
# MODEL_NAME = "MoritzLaurer/ernie-m-large-mnli-xnli" # (0.75686 - F1)
# MODEL_NAME = "microsoft/deberta-xlarge-mnli" # Toi bi out of memroy cuda => thi phai lam sao
# MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli" # (0.7625 - F1)
# MODE_NAME = "uitnlp/CafeBERT"
# MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" # (Co ve thap - 0.7281 - F1)


""":
* Nhóm XLM-RoBERTa / InfoXLM / ERNIE-M :  LR: 1e-5 => 3e-5
- joeddav/xlm-roberta-large-xnli
- FacebookAI/xlm-roberta-large
- microsoft/infoxlm-large
- MoritzLaurer/ernie-m-large-mnli-xnli
- MoritzLaurer/mDeBERTa-v3-base-mnli-xnli

* Nhóm DeBERTa (Hiệu năng cao, cần tối ưu bộ nhớ): LR: 5e-6 => 1.5e-5
- microsoft/deberta-xlarge-mnli
- MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli

* uitnlp/CafeBERT: 2e-5 => 5e-5

* SemViQA/* (Các model đã được fine-tuned): 5e-6 => 2e-5
- SemViQA/qatc-vimrc-viwikifc

- SemViQA/qatc-vimrc-isedsc01

- SemViQA/qatc-infoxlm-viwikifc

- SemViQA/qatc-infoxlm-isedsc01
"""


MODEL_NAMES = [
    "joeddav/xlm-roberta-large-xnli",                           # (0.7851 - F1) - 
    "FacebookAI/xlm-roberta-large",                             # (0. 78192 - F1)
    "microsoft/infoxlm-large",                                  #  (0.7740 - F1)
    "MoritzLaurer/ernie-m-large-mnli-xnli",                     # (0.75686 - F1)
    "microsoft/deberta-xlarge-mnli",
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", # (0.7625 - F1)
    "uitnlp/CafeBERT",
    "SemViQA/qatc-vimrc-viwikifc",    # Thu con nay
    "SemViQA/qatc-vimrc-isedsc01",
    "SemViQA/qatc-infoxlm-viwikifc", # Thu con nay
    "SemViQA/qatc-infoxlm-isedsc01",
]

# backbone tối ưu cho tiếng Việt (MRC-oriented) → xử lý ngữ nghĩa tiếng Việt tốt hơn InfoXLM đa ngôn ngữ trong nhiều trường hợp. viwikifc = fine-tuned trên data dạng Wiki/Fact-check → domain match cao với context bạn đưa (nhiều đoạn giống bài wiki/encyclo).
MODEL_NAME = MODEL_NAMES[3]  

MODEL_OUTPUT_DIR = os.path.join(ROOT_DIR, "models", f"{MODEL_NAME.split('/')[-1]}-tuned")

# --- Cấu hình Tokenizer ---
MAX_LENGTH = 512

# --- Cấu hình Huấn luyện ---
EPOCHS = 10
BATCH_SIZE = 2

EPSILON                     = 1e-8
WEIGHT_DECAY                = 0.02
RANDOM_STATE                = 42
LEARNING_RATE               = 3e-5
PATIENCE_LIMIT              = 2
TOTAL_STEP_SCALE            = 0.1  # Sử dụng số bước để warm-up
CLASSIFIER_DROPOUT          = 0.2
VALIDATION_SPLIT_SIZE       = 0.2
GRADIENT_ACCUMULATION_STEPS = 2     # Tăng dần để phù hợp với GPU VRAM
LABEL_SMOOTHING             = 0.05     # Thêm để regularize và tránh overfitting
    
# --- Ánh xạ Nhãn ---
LABEL_MAP = {'intrinsic': 0, 'extrinsic': 1, 'no': 2}  # contradiction/neutral/entailment
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

# << THÊM DÒNG NÀY (sử dụng con số bạn tính được từ EDA)
# CLASS_WEIGHTS = [1.0393466963622866, 1.0114145354717525, 0.9531590413943355]
CLASS_WEIGHTS = None


config_vars = [
    'MODEL_NAME',
    'MAX_LENGTH',
    'EPOCHS',
    'BATCH_SIZE',
    'GRADIENT_ACCUMULATION_STEPS',
    'LEARNING_RATE',
    'EPSILON',
    'WEIGHT_DECAY',
    'RANDOM_STATE',
    'PATIENCE_LIMIT',
    'TOTAL_STEP_SCALE',
    'CLASSIFIER_DROPOUT',
    'LABEL_SMOOTHING',
    'CLASS_WEIGHTS',
    'LABEL_MAP',
    'ID2LABEL',
]

