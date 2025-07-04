document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const resultDiv = document.getElementById('result');
    let previewImage = null;

    // Hiển thị ảnh ngay khi chọn tệp
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        resultDiv.innerHTML = ''; // Xóa nội dung cũ
        if (file) {
            previewImage = document.createElement('img');
            previewImage.src = URL.createObjectURL(file);
            previewImage.className = 'preview-image';
            previewImage.style.width = '300px'; // Đặt kích thước cố định
            previewImage.style.height = '300px'; // Đặt kích thước cố định
            resultDiv.appendChild(previewImage);
        }
    });

    // Xử lý gửi form và hiển thị kết quả dự đoán
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(form);
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (response.ok) {
                // Thêm kết quả dự đoán bên dưới ảnh
                if (previewImage) {
                    resultDiv.innerHTML = ''; // Xóa nội dung cũ để sắp xếp lại
                    resultDiv.appendChild(previewImage); // Giữ ảnh
                    resultDiv.innerHTML += `
                        <div class="prediction-result">
                            <h3>Kết quả dự đoán</h3>
                            <p>Tệp: ${formData.get('file').name}</p>
                            <p>Nhãn: ${data.predicted_class}</p>
                            <p>Độ tin cậy: ${(data.confidence * 100).toFixed(2)}%</p>
                            <p>Thời gian suy luận: ${data.inference_time.toFixed(4)} giây</p>
                        </div>
                    `;
                }
            } else {
                resultDiv.innerHTML = `
                    ${previewImage ? `<img src="${previewImage.src}" class="preview-image" style="width: 300px; height: 300px;">` : ''}
                    <p class="error">Lỗi: ${data.error}</p>
                `;
            }
        } catch (error) {
            resultDiv.innerHTML = `
                ${previewImage ? `<img src="${previewImage.src}" class="preview-image" style="width: 300px; height: 300px;">` : ''}
                <p class="error">Lỗi: ${error.message}</p>
            `;
        }
    });
});