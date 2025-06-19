# CS317 Lab 1 - MLOps
## Danh sách thành viên:
| Họ và tên      | MSSV | Lớp     |
| :----:        |    :----:   |          :----: |
| [Phạm Huỳnh Nhật Tân](https://github.com/tanphn?tab=repositories)      | 22521309       | CS317.P21  |
| [Phạm Nguyễn Anh Tuấn](https://github.com/nguoimay1103?tab=repositories)   | 22521610        | CS317.P21     |
| [Nguyễn Dương Quốc Thắng](https://github.com/solohito?tab=repositories)   | 22521332       | CS317.P21     |
| [Ngô Nguyễn Nam Trung](https://github.com/namtrunguit?tab=repositories)   | 22521559      | CS317.P21     |
## Mô Tả
Đây là project MLOps Lab 1, nơi triển khai pipeline cho bài toán phân loại hình ảnh với sử dụng các công cụ như Optuna, ClearML, Neptune, DVC và nhiều công cụ khác. Các bước chính của pipeline bao gồm:
- Tiền xử lý dữ liệu
- Huấn luyện mô hình với Optuna để tối ưu hóa hyperparameters
- Sử dụng ClearML để theo dõi quá trình huấn luyện và các kết quả
- Sử dụng Neptune để ghi nhận các thông số và kết quả
- Sử dụng DVC cho quản lý dữ liệu
## Các Công Cụ Sử Dụng
- **Optuna**: Dùng để tối ưu hóa hyperparameters của mô hình.
- **ClearML**: Dùng để theo dõi và quản lý các experiment trong quá trình huấn luyện.
- **Neptune**: Ghi nhận các thông số, mô hình, và kết quả huấn luyện.
- **DVC**: Quản lý dữ liệu và giúp version hóa dữ liệu huấn luyện.
- **PyTorch & Torchvision**: Dùng để xây dựng và huấn luyện mô hình học sâu.
- **Scikit-learn**: Dùng để đánh giá mô hình và tính toán các chỉ số như accuracy.
## 📁 Cấu Trúc Pipeline
- `data/`
  - `raw/`: 📦 Dữ liệu gốc (DVC quản lý)
  - `processed/`: 🧹 Dữ liệu đã xử lý (DVC quản lý)
- `src/` — 💻 Mã nguồn chính:
  - `data_preprocessing.py`: 🌸 Tiền xử lý dữ liệu (Stage: preprocess)
  - `train.py`: 🧠 Huấn luyện mô hình
    - 🎯 Optuna: Tối ưu siêu tham số
    - 🛠️ ClearML: Theo dõi và orchestration
    - 📊 Neptune: Log mô hình và metrics
  - `evaluate.py`: 📈 Đánh giá mô hình (Stage: evaluate)
  - `tune.py`: 🔁 Chạy tuning riêng
- `models/checkpoints/`: 💾 Lưu checkpoints
- `metrics/`: 📄 File đánh giá (.json/.csv — DVC quản lý)
- `dvc.yaml`: ⚙️ Định nghĩa pipeline (preprocess → train → evaluate)
- `clearml.conf`: 🔧 Cấu hình ClearML
- `requirements.txt`: 📦 Thư viện cần thiết
## Cài Đặt Môi Trường
1. **Clone repository** về máy:
   ```bash
   git clone https://github.com/tanphn/cs317_lab1.git
   cd cs317_lab1
2. **Cài đặt thư viện cần thiết**
   ```bash
   pip install -r requirements.txt
File requirements.txt bao gồm các thư viện chính như:
- **torch**: Thư viện học sâu PyTorch.
- **torchvision**: Hỗ trợ xử lý hình ảnh với PyTorch.
- **optuna**: Tối ưu hóa hyperparameters.
- **clearml**: Quản lý pipeline và thí nghiệm.
- **neptune**: Log metric và mô hình.
- **dvc**: Quản lý dữ liệu.
- **scikit-learn**: Đánh giá mô hình.
- **mlflow, fastapi, uvicorn, prometheus-client, pytest**: Các công cụ bổ sung cho MLOps.
3. **Cấu hình dvc**
   ```bash
   dvc init
  - Nếu không chạy được thì tải extension dvc ở trên visual studio.
## Chạy Pipeline

### Chạy Pipeline với DVC

Pipeline được định nghĩa trong file `dvc.yaml` và bao gồm 3 bước:

- **Preprocess**: Xử lý dữ liệu từ `data/raw` và lưu vào `data/processed`.
- **Train**: Huấn luyện mô hình ResNet18 trong 10 epoch, tối ưu `lr` và `batch_size` bằng Optuna.
- **Evaluate**: Đánh giá mô hình và lưu kết quả vào `metrics/eval.json`.

Chạy toàn bộ pipeline:

```bash
dvc repro
```

---

### Chạy Pipeline với ClearML

Bạn cũng có thể chạy pipeline thông qua **ClearML** bằng lệnh:

```bash
python src/clearml_pipeline.py
```

ClearML sẽ thực hiện toàn bộ pipeline theo đúng logic trong `dvc.yaml`, đồng thời log toàn bộ quá trình lên ClearML Dashboard.

---

## **Theo dõi kết quả**
- **ClearML**:  
  Sau khi chạy pipeline, bạn sẽ thấy link ClearML results page trong log (ví dụ: `https://app.clear.ml/projects/...`).  
  Truy cập link để xem trạng thái pipeline, log, và artifact (như checkpoint mô hình, file metric).  
- **Neptune**:  
  Truy cập `app.neptune.ai` để xem metric (loss, accuracy), checkpoint mô hình, và hyperparameters.  
- **DVC**:  
  Dữ liệu được quản lý trong thư mục `data/processed`. Dùng lệnh sau để kéo dữ liệu mới nhất nếu cần:  
  ```bash
  dvc pull
## **Video chạy pineline**
  Video được lưu ở thư mục: **https://drive.google.com/drive/folders/1HiKB9ii385YYsoV-60tHD3TeShuuaiwW?usp=drive_link**
