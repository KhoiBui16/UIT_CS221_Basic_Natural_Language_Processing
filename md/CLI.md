# Hướng dẫn các lệnh hay dùng (user guest này không có quyền sudo -> cần download gì thì alo khoibui)

- Có cài `docker`, `git`, `vim`, `nvim`

> Nhớ đổi tên notebook_name

## 0. Upload và download file từ server về local thông qua ssh
### +) kiểm tra ip sau khi kết nối với tailscale (phải yêu cầu add vào vpn tailscale qua gmail mới truy cập ip này được)
```bash
# coi phiên bản lix -> ra ip
tailscale status
```

### +) Cách nối nối qua SSH
```bash
# dùng ip của tailscale để kết nối 
## Đổi tên: user_name -> guest
## Đổi tên: ip_tailscale -> ip sau khi chạy tailscale status
ssh user_name@**ip_tailscale**

# sau đó nhập password cho user_name
```

### +) Cách donwload files/folders từ server về local (ngược lại) qua SSH
#### Download 1 file (server → local):
```bash
# mode: đọc các hướng dẫn dưới và thay thế chữ mode
scp user@server:/path/to/remote/file.txt /local/path/

```

#### +) Upload 1 file (local → server):
```bash
scp /local/path/file.txt user_name@server:/path/to/remote/

```
#### +) Copy thư mục (recursive):
```bash
scp -r user@server:/path/to/remote/folder /local/path/

# hoặc upload:
scp -r /local/folder user@server:/path/to/remote/

```

|             Tùy chọn | Ý nghĩa ngắn                                         | Ví dụ (thay giá trị cho phù hợp)                              |
| -------------------: | ---------------------------------------------------- | ------------------------------------------------------------- |
|                 `-r` | Sao chép đệ quy — copy thư mục cùng toàn bộ file con | `scp -r user@host:/remote/dir /local/dir`                     |
|          `-P <port>` | Cổng SSH (chú ý: chữ **P** viết hoa)                 | `scp -P 2222 file.txt user@host:/path/`                       |
| `-i <identity_file>` | Dùng private key (thay vì mật khẩu)                  | `scp -i ~/.ssh/id_rsa file.txt user@host:/path/`              |
|                 `-p` | Preserve — giữ nguyên thời gian sửa/perm của file    | `scp -p file.txt user@host:/path/`                            |
|                 `-C` | Bật nén khi truyền (tốt cho nhiều file nhỏ)          | `scp -C bigdir.tar.gz user@host:/path/`                       |
|         `-l <limit>` | Giới hạn băng thông (kbit/s)                         | `scp -l 500 file.bin user@host:/path/`                        |
|                 `-v` | Verbose — in thông tin debug của SSH/scp             | `scp -v file.txt user@host:/path/`                            |
|                 `-q` | Quiet — giảm output                                  | `scp -q file.txt user@host:/path/`                            |
|  `-o "<SSH option>"` | Chuyển option cho `ssh` (ví dụ kiểm soát host key)   | `scp -o "StrictHostKeyChecking=no" file.txt user@host:/path/` |
|   `-S <ssh_program>` | Dùng chương trình `ssh` khác                         | `scp -S /usr/bin/ssh file.txt user@host:/path/`               |


---
## 1. Chạy nền jupyter notebook và gọi port mỗi lần chạy
### +) vừa chạy nền notebook & real-time terminal + log => Dùng papermill(phải download cho từng env)
```bash
mkdir -p logger
stdbuf -oL papermill notebook_name.ipynb notebook_name_out.ipynb --log-output | tee logger/run.log
```

### +) dùng để xem log chạy cùng lúc
```bash
tail -f run.log 
```

### +) Cấp quyền cho file sh
```bash
chmod +x run_with_port.sh
```

### +) chạy sh để tự động tìm port (khi gọi ollama) và chạy file nền tmux từ papermill ở câu lệnh tích hợp trên từ file run_with_port.sh
```bash
./run_with_port.sh notebook_name.ipynb notebook_name_out.ipynb
```

---
## 2. Miniconda (nên tải bằng conda install cho đỡ xung độ => Nhưng cứ cài pip install đi cho chung version)
### +) Tạo env
```bash
# Nhớ đổi: env_name và python_version
conda create -n env_name python=python_version
```

### +) Xóa env
```bash
# Nhớ đổi: env_name
conda remove -n env_name --all
```

### +) Xem danh sách env
```bash
# chọn 1 trong 2 lệnh dưới
conda env list

conda info --envs
```

### +) Kích hoạt env
```bash
conda activate env_name
```

### +) Hủy env
```bash
conda deactivate
```

### +) Nếu tải pytorch hay tensorflow => phải coi (CUDA version) hiện tại
```bash
nvidia-smi

# dùng để coi cấu hình gpu và phiên bản theo từng giây
watch -n num_second nvidia-smi
```

### +) Tải package
```bash
conda install
```

```bash
pip install
```

```bash
# tải từ file requirements.txt
pip install -r requirements.txt
```

----
## 3. Chạy nền cho cả file python và notebook
- Phải tạo ra tmux session và sau đó nhớ activate env trong tmux

### +) Tạo tmux session
```bash
tmux new -s session_name
```

### +) Liệt kê tmux session
```bash
tmux ls
```

### +) Truy cập vào tmux session
```bash
tmux attach -t session_name
```

### +) Để thoát khỏi tmux session hiện tại mà không hủy session
```bash
# không nhấn cùng lúc -> làm xong Ctrl + b rồi mới nhấn d
Ctrl + b + d
```

### +) Xóa tmux session
```bash
# Ở trong tmux session, nhấn phím sau:
exit
```

## 4. Quản lý file và thư mục
### +) Liệt kê file
```bash
# Liệt kê thông thường:
ls

# Liệt kê xem quyền:
ls -l

# Liệt kê đầy đủ cả file ẩn:
ls -la
```

### +) Các lệnh thao tác nén và giải nén file
#### 1) zip
```bash
## 1. zip
zip [options] tên_file_zip.zip file1 file2 thư_mục1 thư_mục2 ...
```
- Các tuỳ chọn thường dùng:

    ---
    | Tuỳ chọn | Ý nghĩa                                                                          |
    | -------- | -------------------------------------------------------------------------------- |
    | `-r`     | Nén đệ quy thư mục (bao gồm tất cả file con và thư mục con)                      |
    | `-u`     | Update: thêm file mới vào zip nếu file đó mới hơn bản trong zip hoặc nếu chưa có |
    | `-m`     | Move: nén file/thư mục và **xóa** file/thư mục gốc sau khi nén xong              |
    | `-x`     | Loại trừ các file/phần cụ thể khỏi trong khi nén                                 |

        

- Ví dụ:

    1> Nén 1 file:
    ```bash
    zip myarchive.zip file1.txt
    ```

    2> Nén nhiều file:
    ```bash
    zip myarchive.zip file1.txt file2.txt dir1/
    ```

    3> Nén toàn bộ thư mục:
    ```bash
    zip -r folder_archive.zip my_folder/
    ```

    4> Nén nhưng loại trừ file .log:
    ```bash
    zip -r myarchive.zip my_folder/ -x "*.log"
    ```
---
#### 2) unzip
```bash
unzip tên_file.zip
```

- Các tuỳ chọn thường dùng:

    ---

    | Tuỳ chọn         | Ý nghĩa                                                                                      |
    | ---------------- | -------------------------------------------------------------------------------------------- |
    | `-d <directory>` | Giải nén vào thư mục chỉ định                                                                |
    | `-l`             | Liệt kê các file bên trong archive mà không giải nén  |
    | `-t`             | Kiểm tra tính hợp lệ của file zip (test)                                   |
    | `-o`             | Ghi đè file đã có mà không hỏi (overwrite)                                 |
    | `-n`             | Không ghi đè file nếu đã có (skip existing)                                |
    | `-q`             | Chế độ “quiet” – ít thông báo hơn khi thực thi                            |
    | `-x`             | Loại trừ file cụ thể khi giải nén                                          |


- Ví dụ:

    1> Giải nén vào thư mục hiện tại:
    ```bash
    unzip myarchive.zip
    ```

    2> Giải nén vào thư mục khác:
    ```bash
    unzip myarchive.zip -d /path/to/destination_folder
    ```

    3> Chỉ liệt kê nội dung zip mà không giải nén:
    ```bash
    unzip -l myarchive.zip
    ```

    4> Kiểm tra file zip:
    ```
    unzip -t myarchive.zip
    ```

    5> Giải nén và ghi đè file nếu có:
    ```bash
    unzip -o myarchive.zip
    ```

    6> Giải nén nhưng không ghi đè nếu file đã tồn tại:
    ```bash
    unzip -n myarchive.zip
    ```

    7> Giải nhiều file zip cùng lúc:
    ```bash
    unzip '*.zip'
    ```
---

#### 3) tar
Định dạng: `.tar.gz / .tgz`:

```bash
tar -xzvf file.tar.gz
```

- trong đó:

    > -x extract

    > -z dùng gzip

    > -v verbose (hiển thị các file được giải)

    > -f chỉ rõ file

## 5. Xem dung lượng folders files
### +) Coi tổng quát bằng phần mền GUI
```bash
ncdu /home/guest/
```

### +) du (disk usage)
```bash
# Hiển thị tổng dung lượng của folder /path/to/folder theo định dạng dễ đọc (human-readable: K, M, G)
du -sh /path/to/folder
```

```bash
# Hiển thị dung lượng các thư mục con trong folder đó, chỉ sâu 1 cấp => có thể thay thế số 1 theo số cấp muốn coi
du -h --max-depth=1 /path/to/folder
```

```bash
# Xem kích thước từng thư mục/file con trong folder hiện tại (không bao gồm folder gốc).
du -sh *
```

### +) df (disk free)
```bash
# Hiển thị dung lượng tổng / đã dùng / còn trống của các phân vùng (filesystem) với đơn vị dễ hiểu.
df -h
```

```bash
# Thêm cột cho biết loại filesystem của từng phân vùng
df -T
```

```bash
# Xem riêng phân vùng/mount point cụ thể
df -h /some/mount/point
```

### +) Kết hợp du + sort + head để tìm thư mục / file lớn nhất
```bash
# sort -h: sắp xếp theo kích thước human-readable (ví dụ: 1K, 200M, 2G
du -h /path/to/folder | sort -h
```

```bash
# Dòng trên hiển thị 5 thư mục (hoặc file) con lớn nhất trong folder /path/to/folder
du -sh /path/to/folder/* | sort -hr | head -n 5
```