# app/run_app.py
import subprocess
import time
import os
import sys
import signal
import socket

# --- CẤU HÌNH PORT MỚI ---
BACKEND_PORT = 8095
FRONTEND_PORT = 8505
# -------------------------

fastapi_process = None
streamlit_process = None


def is_port_open(port):
    """Kiểm tra xem port có đang mở (có tiến trình lắng nghe) không"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)  # Check nhanh
        return s.connect_ex(("localhost", port)) == 0


def kill_port_processes(port):
    """Tìm và diệt tất cả tiến trình đang chiếm port bằng fuser."""
    if is_port_open(port):
        print(f"🧹 Đang dọn dẹp port {port}...")
        # Gửi tín hiệu SIGKILL (-9) để diệt ngay lập tức
        os.system(f"fuser -k -9 {port}/tcp > /dev/null 2>&1")


def wait_until_port_free(port, timeout=10):
    """
    HÀM QUAN TRỌNG: Chờ cho đến khi port thực sự được giải phóng.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if not is_port_open(port):
            print(f"✅ Port {port} đã rảnh.")
            return True
        print(f"⏳ Port {port} vẫn đang bận, chờ 0.5s...")
        time.sleep(0.5)
    print(f"❌ LỖI: Port {port} vẫn bị chiếm dụng sau {timeout}s.")
    return False


def cleanup_and_verify():
    """Quy trình dọn dẹp và kiểm tra nghiêm ngặt."""
    print("\n🧹 Bắt đầu dọn dẹp hệ thống...")

    # 1. Gửi lệnh terminate cho các process con
    if fastapi_process:
        fastapi_process.terminate()
    if streamlit_process:
        streamlit_process.terminate()

    # --- FIX: Diệt theo TÊN TIẾN TRÌNH để giết các "zombie" ---
    os.system("pkill -9 -f 'uvicorn app.app_fastapi'")
    os.system("pkill -9 -f 'streamlit run app/app_streamlit_local.py'")
    # --------------------------------------------------------

    # 2. Diệt tận gốc các port
    kill_port_processes(BACKEND_PORT)
    kill_port_processes(FRONTEND_PORT)

    # 3. Chờ xác nhận từ hệ điều hành là port đã rảnh
    print("⏳ Đang đợi OS giải phóng ports...")
    be_free = wait_until_port_free(BACKEND_PORT)
    fe_free = wait_until_port_free(FRONTEND_PORT)

    if be_free and fe_free:
        print("✅ Hệ thống đã sạch sẽ.")
        return True
    else:
        print("❌ LỖI: Không thể giải phóng port. Vui lòng kiểm tra thủ công.")
        return False


def signal_handler(sig, frame):
    print("\n🛑 Đang dừng hệ thống...")
    cleanup_and_verify()
    sys.exit(0)


def main():
    global fastapi_process, streamlit_process
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("🚀 Đang khởi động hệ thống (Local Client-Server)...")

    # 1. Dọn dẹp sạch sẽ và CHỜ ĐỢI
    if not cleanup_and_verify():
        print("Vui lòng thử lại sau vài giây.")
        sys.exit(1)  # Thoát nếu không thể dọn dẹp port

    # 2. Khởi chạy Backend
    print(f"⏳ Đang bật Backend (FastAPI) trên port {BACKEND_PORT}...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    env["PYTHONUNBUFFERED"] = "1"

    fastapi_process = subprocess.Popen(
        [
            "uvicorn",
            "app.app_fastapi:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(BACKEND_PORT),
            # --- ĐÃ XÓA "--reload" ĐỂ TRÁNH XUNG ĐỘT ---
        ],
        env=env,
    )

    # Đợi Backend sẵn sàng
    print("⏳ Đang đợi Backend khởi động...")
    ready = False
    for i in range(30):  # Đợi tối đa 30s
        if is_port_open(BACKEND_PORT):
            print(f"✅ Backend đã online sau {i+1}s!")
            ready = True
            break
        time.sleep(1)

    if not ready:
        print("❌ Backend khởi động thất bại hoặc quá lâu. Vui lòng kiểm tra log.")

    # 3. Khởi chạy Frontend (Local version)
    print(f"⏳ Đang bật Frontend (Streamlit Local) trên port {FRONTEND_PORT}...")
    streamlit_process = subprocess.Popen(
        [
            "streamlit",
            "run",
            "app/app_streamlit_local.py",  # <-- Đảm bảo tên file này chính xác
            "--server.port",
            str(FRONTEND_PORT),
            "--server.address",
            "localhost",
            # --- TẮT CƠ CHẾ RELOAD CỦA STREAMLIT ---
            "--server.fileWatcherType",
            "none",
            "--server.runOnSave",
            "false",
        ],
        env=env,
    )

    print("\n✨ --- HỆ THỐNG SẴN SÀNG --- ✨")
    print(f"👉 Frontend UI: http://localhost:{FRONTEND_PORT}")
    print(f"👉 Backend API: http://localhost:{BACKEND_PORT}/docs")
    print("-----------------------------------")
    print("Nhấn [Ctrl+C] để dừng tất cả.\n")

    try:
        # Vòng lặp giám sát
        while True:
            if fastapi_process.poll() is not None:
                print("\n❌ Backend (FastAPI) đã dừng đột ngột!")
                raise KeyboardInterrupt
            if streamlit_process.poll() is not None:
                print("\n❌ Frontend (Streamlit) đã dừng đột ngột!")
                raise KeyboardInterrupt
            time.sleep(2)

    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()
