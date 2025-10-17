💡 Tóm tắt quy trình chuẩn hoá môi trường cho project AI/DL
Bước 1 – Export phần Conda
conda env export --from-history > environment.yml


👉 Lệnh này chỉ lưu các gói bạn cài bằng Conda (Python, pip, v.v.), không chứa thư viện pip.

Bước 2 – Export phần Pip
pip freeze > requirements.txt


👉 File này chứa toàn bộ các gói Python (Transformers, Torch, SentenceTransformers, …) cùng phiên bản chính xác.

Bước 3 – Gộp hai phần lại (chỉnh file environment.yml)

Mở file environment.yml vừa tạo, thêm phần pip vào cuối cùng như sau 👇

name: cs221
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - setuptools
  - wheel
  - pip:
      - -r requirements.txt


Nghĩa là: Conda cài phần lõi (Python + pip), sau đó pip đọc requirements.txt để cài phần còn lại.

Bước 4 – Kiểm tra tái tạo môi trường local

Trên máy khác hoặc môi trường sạch:

conda env create -f environment.yml


✅ Nếu lệnh này chạy xong không lỗi → bạn đã tái tạo được môi trường 100%.

Bước 5 – Dùng cho Docker

Tạo file Dockerfile (tại cùng thư mục với 2 file trên):

FROM continuumio/miniconda3

WORKDIR /app

# Copy environment definition
COPY environment.yml .
COPY requirements.txt .

# Tạo conda env
RUN conda env create -f environment.yml

# Kích hoạt env mặc định cho shell
SHELL ["conda", "run", "-n", "cs221", "/bin/bash", "-c"]

# Copy source code
COPY . .

# Entrypoint
CMD ["python", "main.py"]


Build Docker image:

docker build -t cs221 .


Chạy container (nếu có GPU):

docker run --gpus all -it cs221

🎯 Tóm tắt nhanh
Bước	Lệnh / Hành động	Mục đích
1	conda env export --from-history > environment.yml	Lưu gói Conda
2	pip freeze > requirements.txt	Lưu gói Pip
3	Chỉnh environment.yml thêm phần - pip: - -r requirements.txt	Gộp 2 phần
4	conda env create -f environment.yml	Tái tạo môi trường local
5	Tạo Dockerfile (sử dụng file env)	Tạo image tái lập môi trường

👉 Sau bước này, bạn có 3 file quan trọng:

environment.yml
requirements.txt
Dockerfile


→ chỉ cần gửi 3 file + code project là bất kỳ ai (hoặc server nào) đều có thể chạy được đúng phiên bản và CUDA.