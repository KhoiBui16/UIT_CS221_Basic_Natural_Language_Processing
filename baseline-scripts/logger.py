# logger.py

import logging
import os
from datetime import datetime

# Thư mục gốc để lưu tất cả các file log
LOG_BASE_DIR = "logs"

# Dùng một dictionary để lưu các logger đã tạo, tránh việc tạo lại và gây ra log trùng lặp
_loggers = {}

def setup_logger(model_name: str, log_level=logging.INFO):
    """
    Thiết lập và trả về một logger để ghi log vào cả console và file.

    - Mỗi model sẽ có một thư mục log riêng dựa trên `model_name`.
    - Mỗi lần chạy sẽ tạo một file log mới có tên là timestamp (ví dụ: 2023-10-27_15-30-00.log).
    - Đảm bảo không có log nào bị ghi đè.

    Args:
        model_name (str): Tên của model, dùng để tạo thư mục con. 
                          Ví dụ: 'xnli-large-tuned'.
        log_level (int): Cấp độ log, mặc định là logging.INFO.

    Returns:
        logging.Logger: Instance của logger đã được cấu hình.
    """
    # Nếu logger cho model này đã tồn tại, trả về nó ngay lập tức
    if model_name in _loggers:
        return _loggers[model_name]

    # Xử lý tên model để an toàn khi tạo tên thư mục (thay thế "/")
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    model_log_dir = os.path.join(LOG_BASE_DIR, safe_model_name)
    os.makedirs(model_log_dir, exist_ok=True)

    # Tạo logger
    logger = logging.getLogger(safe_model_name)
    logger.setLevel(log_level)
    
    # Ngăn không cho log lan truyền đến root logger để tránh in ra console 2 lần
    logger.propagate = False

    # Định dạng cho log message
    formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Tạo File Handler để ghi log ra file
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file_path = os.path.join(model_log_dir, f"{timestamp}.log")
    
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Tạo Console (Stream) Handler để in log ra màn hình
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Thêm các handler vào logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Lưu logger vào cache
    _loggers[model_name] = logger
    
    logger.info(f"Logger cho '{safe_model_name}' đã được khởi tạo. File log: {log_file_path}")
    
    return logger