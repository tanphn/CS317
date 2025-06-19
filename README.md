
<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: 5;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<!-- Title -->
<h1 align="center"><b>CS317.P21 - PHÁT TRIỂN VÀ VẬN HÀNH HỆ THỐNG MÁY HỌC</b></h1>

## Môn học 
<a name="gioithieumonhoc"></a>
* *Môn học*: Phát triển và vận hành hệ thống máy học
* *Mã lớp*: CS317.P21
* *Year*: 2024-2025
## Giáo viên
<a name="giangvien"></a>
* *Đỗ Văn Tiến* - tiendv@uit.edu.vn
* *Lê Trần Trọng Khiêm* - khiemltt@uit.edu.vn

# CS317 - Emotion Recognition System with End-to-End MLOps Pipeline Deployment
## Danh sách thành viên:
| Họ và tên      | MSSV | Lớp     |
| :----:        |    :----:   |          :----: |
| [Phạm Huỳnh Nhật Tân](https://github.com/tanphn?tab=repositories)      | 22521309       | CS317.P21  |
| [Phạm Nguyễn Anh Tuấn](https://github.com/nguoimay1103?tab=repositories)   | 22521610        | CS317.P21     |
| [Nguyễn Dương Quốc Thắng](https://github.com/solohito?tab=repositories)   | 22521332       | CS317.P21     |
| [Ngô Nguyễn Nam Trung](https://github.com/namtrunguit?tab=repositories)   | 22521559      | CS317.P21     |
## Mô Tả
Đây là một dự án xây dựng hệ thống nhận diện cảm xúc khuôn mặt (Facial Emotion Recognition - FER) từ đầu đến cuối, tích hợp học sâu với quy trình MLOps hiện đại. Hệ thống bao gồm các giai đoạn từ huấn luyện mô hình, triển khai API, đến giám sát và cảnh báo theo thời gian thực.
## 🧠 Mục tiêu

- Xây dựng mô hình học sâu nhận diện cảm xúc từ ảnh khuôn mặt.
- Triển khai mô hình thành API sử dụng FastAPI và Docker.
- Tích hợp hệ thống giám sát và cảnh báo để theo dõi hoạt động và hiệu năng của mô hình.
- Tối ưu hóa quy trình phát triển mô hình với các công cụ MLOps.

## 🛠️ Công nghệ sử dụng

| Thành phần       | Công cụ/Thư viện                           |
|------------------|--------------------------------------------|
| Huấn luyện mô hình | PyTorch, ResNet18, Optuna, ClearML, Neptune AI |
| Triển khai API     | FastAPI, Docker                           |
| Giám sát hệ thống  | Prometheus, Grafana, Fluentd, Alertmanager |

## 🧪 Dataset

- **Tên:** FER-2013 (Facial Expression Recognition 2013)
- **Số lượng ảnh:** ~7000 ảnh grayscale kích thước 48x48 pixels
- **Nhãn cảm xúc:** Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

## 🔄 Pipeline MLOps

### Giai đoạn Huấn luyện
Quá trình huấn luyện mô hình được thiết kế dưới dạng pipeline với các bước rõ ràng và tự động hóa, giúp đảm bảo tính nhất quán và khả năng mở rộng của hệ thống.

#### 1. Tiền xử lý dữ liệu

- Bộ dữ liệu **FER-2013** bao gồm khoảng 7000 ảnh grayscale kích thước 48x48 pixels.
- Mỗi ảnh được gán nhãn với một trong **7 cảm xúc**:  
  `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`
- Dữ liệu được:
  - Chuẩn hóa kích thước.
  - Chuyển sang định dạng tensor cho PyTorch.
  - Chia thành tập huấn luyện và kiểm tra (train/test split).
- Áp dụng các kỹ thuật tăng cường dữ liệu (*data augmentation*) như:
  - Xoay ảnh
  - Lật ngang
  - Thay đổi độ sáng nhẹ  
  → Nhằm tăng khả năng tổng quát của mô hình.

#### 2. Huấn luyện mô hình ResNet18

- Mô hình sử dụng kiến trúc **ResNet18** – một mạng CNN phổ biến trong các bài toán thị giác máy tính.
- Có thể khởi tạo từ đầu hoặc dùng trọng số pre-trained.
- Các cấu hình chính:
  - Hàm mất mát: `CrossEntropyLoss`
  - Optimizer: `Adam` hoặc `SGD` tùy theo kết quả tuning
  - Epoch: 10 đến 30 (tùy cấu hình)

#### 3. Tối ưu siêu tham số bằng Optuna

- **Optuna** được sử dụng để tìm kiếm tự động các siêu tham số tốt nhất.
- Tổng số trial: ~50
- Các siêu tham số được tối ưu:
  - `Learning rate`: từ `0.0001` đến `0.01`
  - `Batch size`: `32`, `64`, `128`
  - `Optimizer`: `Adam`, `SGD`
- Tiêu chí đánh giá: độ chính xác trên tập validation (`val_accuracy`)

#### 4. Theo dõi quá trình huấn luyện

##### 📘 ClearML
- Tự động ghi log toàn bộ pipeline:
- Cho phép theo dõi trực tuyến qua ClearML Dashboard

##### 📙 Neptune AI
- Ghi nhận chỉ số như:
  - `loss`, `accuracy`
  - `confusion matrix`
  - `learning rate scheduler`
- Hỗ trợ trực quan hóa thời gian thực trên nền tảng web

---

#### 🔍 Kết quả tuning tiêu biểu

- **Learning rate** tốt nhất: `0.001`  
- **Batch size** tốt nhất: `64`  
- **Optimizer**: `Adam`  
- **Validation Accuracy**: **65%**

### Giai đoạn Triển khai
Sau khi mô hình được huấn luyện và đánh giá, bước tiếp theo là triển khai hệ thống thành một dịch vụ API để phục vụ các ứng dụng thực tế. Quá trình triển khai sử dụng **FastAPI** để xây dựng API và **Docker** để đóng gói ứng dụng, đảm bảo tính di động và nhất quán.

### Quy trình Triển khai

#### 1. Xây dựng API Dự đoán

Ứng dụng dự đoán được phát triển bằng **FastAPI**, cho phép nhận ảnh khuôn mặt từ người dùng, xử lý và trả về kết quả dự đoán cảm xúc. API bao gồm hai endpoint chính:

- **`POST /predict`**: Nhận ảnh đầu vào, thực hiện suy luận và trả về nhãn cảm xúc cùng độ tin cậy.
- **`GET /metrics`**: Cung cấp các chỉ số hiệu suất hệ thống (ví dụ: số lượng yêu cầu, thời gian phản hồi) để phục vụ giám sát.

**Ví dụ Kết quả Trả về** cho `/predict`:
```json
{
  "emotion": "Vui vẻ",
  "confidence": 0.91
}
```
#### 2. Đóng gói Ứng dụng với Docker
Toàn bộ mã nguồn, mô hình `.pt` đã huấn luyện và các thư viện cần thiết được đóng gói vào một Docker image. Việc sử dụng Docker giúp đảm bảo ứng dụng hoạt động nhất quán trên mọi môi trường, từ máy tính cá nhân đến các hệ thống cloud như AWS, GCP, Azure.

Đầu tiên, một `Dockerfile` được viết để mô tả quá trình xây dựng môi trường bao gồm cài đặt Python, các thư viện cần thiết (từ `requirements.txt`), sao chép mã nguồn và thiết lập điểm khởi chạy ứng dụng bằng FastAPI.

Sau đó, sử dụng lệnh sau để build Docker image:
```bash
docker build -t emotion-api .
```
Khi image đã được tạo thành công, container có thể được khởi chạy bằng lệnh:
```bash
docker run -d -p 8000:8000 emotion-api
```
Sau khi container chạy, API có thể được truy cập tại http://localhost:8000. Người dùng có thể gửi ảnh khuôn mặt thông qua phương thức POST /predict để nhận kết quả dự đoán cảm xúc, hoặc truy cập GET /metrics để lấy thông tin giám sát hiệu suất hệ thống. Việc tích hợp /metrics giúp chuẩn bị sẵn nền tảng để kết nối với các công cụ như Prometheus và Grafana ở giai đoạn giám sát.

Việc đóng gói bằng Docker mang lại nhiều lợi ích như khả năng triển khai nhanh, dễ kiểm soát phiên bản, dễ tái sử dụng và có thể dễ dàng mở rộng quy mô với Kubernetes hoặc các nền tảng container orchestration khác.
### 📊 Giai đoạn Giám sát Hệ thống

Sau khi hệ thống được triển khai thành công dưới dạng API, bước tiếp theo là thiết lập cơ chế giám sát để theo dõi hiệu suất và phát hiện lỗi kịp thời trong quá trình vận hành. Việc giám sát đóng vai trò rất quan trọng trong môi trường thực tế, đặc biệt khi hệ thống phục vụ cho nhiều người dùng hoặc chạy liên tục trong thời gian dài.

Hệ thống giám sát được xây dựng với sự kết hợp của các công cụ: **Prometheus** để thu thập metric, **Grafana** để trực quan hóa dữ liệu, **Fluentd** để thu log và **Alertmanager** để gửi cảnh báo tự động qua email khi phát hiện sự cố.

### Quy trình Thiết lập Giám sát

1. **FastAPI tích hợp `/metrics` endpoint**, cung cấp các chỉ số như:
   - Số lượng request thành công/thất bại
   - Thời gian suy luận trung bình
   - Tỉ lệ lỗi
   Endpoint này được Prometheus truy cập định kỳ để lấy dữ liệu.

2. **Prometheus** được cấu hình để theo dõi API và các thành phần hệ thống khác. Dữ liệu metric thu thập từ FastAPI được lưu trữ tại đây.

3. **Grafana** kết nối với Prometheus để tạo các biểu đồ, dashboard thời gian thực giúp theo dõi trực quan tình trạng hoạt động của hệ thống.

4. **Alertmanager** chịu trách nhiệm gửi cảnh báo qua email khi phát hiện các điều kiện bất thường như API ngừng phản hồi, tỷ lệ lỗi tăng cao hoặc độ tin cậy của mô hình thấp.

---

## 🛠️ Hướng dẫn Sử dụng Hệ thống Giám sát

### 1. Gửi Yêu cầu Dự đoán (FastAPI)

- Truy cập `http://localhost:8000/docs` để mở giao diện Swagger UI của FastAPI.
- Tại đây, bạn có thể thử gửi ảnh khuôn mặt đến endpoint `/predict` và xem kết quả trả về.

### 2. Truy cập Prometheus

- Mở trình duyệt và vào địa chỉ: [http://localhost:9090](http://localhost:9090)
- Vào tab **Status > Targets** để kiểm tra xem các dịch vụ có đang được giám sát (trạng thái `UP`).

### 3. Trực quan hóa Metric với Grafana

- Truy cập `http://localhost:3000`
- Đăng nhập với tài khoản mặc định:  
  - **Username:** `admin`  
  - **Password:** `admin` (bạn có thể được yêu cầu đổi mật khẩu sau lần đăng nhập đầu tiên)
- Cấu hình nguồn dữ liệu Prometheus:
  - Vào **Configuration (biểu tượng bánh răng)** > **Data Sources** > **Add data source**
  - Chọn **Prometheus**, nhập URL: `http://lab3-prometheus:9090`
  - Nhấn **Save & Test**
- Nhập Dashboard có sẵn:
  - Vào **Dashboard** > **New** > **Import**
  - Tải lên file `grafana-dashboard.json` và nhấn **Load**

### 4. Cấu hình Alertmanager để Gửi Cảnh Báo qua Email

Việc gửi email giúp bạn phát hiện sớm các sự cố như API không hoạt động, tỷ lệ lỗi cao hoặc độ chính xác mô hình thấp.

#### Bước 1: Chuẩn bị thông tin email (ví dụ: Gmail)

- **SMTP Host**: `smtp.gmail.com`
- **SMTP Port**: `587` (TLS)
- **Username**: địa chỉ Gmail của bạn
- **Password**: sử dụng App Password (nếu bật xác minh 2 bước)

> Để tạo App Password với Gmail:  
> Truy cập [https://myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords), đăng nhập và tạo mật khẩu ứng dụng.

#### Bước 2: Chỉnh sửa file `alertmanager.yml`

```yaml
global:
  resolve_timeout: 5m
```
route:
  receiver: 'email-notifications'
  group_by: ['alertname']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 12h

receivers:
  - name: 'email-notifications'
    email_configs:
      - to: 'your.email@gmail.com'
        from: 'your.email@gmail.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'your.email@gmail.com'
        auth_password: 'your_app_password'
        require_tls: true
## 📈 Kết quả

- Hệ thống hoạt động ổn định, có thể mở rộng và dễ dàng tích hợp.
- Mô hình đạt độ chính xác tốt trên tập validation.
- Dashboard thời gian thực giúp phát hiện và xử lý lỗi nhanh chóng.

## ⚠️ Hạn chế & Hướng phát triển

### Hạn chế:
- Dataset còn nhỏ, mô hình có thể bị overfitting.
- Một số cảm xúc có thể bị nhầm lẫn.
- Quá trình tuning yêu cầu tài nguyên tính toán lớn (GPU).

### Hướng phát triển:
- Sử dụng các tập dữ liệu lớn hơn và đa dạng hơn.
- Áp dụng kỹ thuật kiểm thử mô hình nâng cao.
- Cải thiện khả năng thích nghi với dữ liệu mới theo thời gian.
