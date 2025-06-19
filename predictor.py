import torch
import time
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torchvision.transforms as transforms
from PIL import Image
import os

# 1. Khởi tạo kiến trúc ResNet18 (6 lớp đầu ra)
model = models.resnet18(weights=None)
num_classes = 6
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# 2. Load trọng số đã huấn luyện
model_path = os.path.join(os.path.dirname(__file__), 'model', 'best_model.pt')
model.load_state_dict(torch.load(model_path, map_location="cpu"))

# 3. Chuyển về chế độ eval để inference
model.eval()

# 4. Tiền xử lý ảnh đầu vào
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. Mapping nhãn
class_names = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
# 6. Hàm dự đoán tên lớp từ ảnh
def predict_image(image: Image.Image):
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        start_time = time.perf_counter()
        output = model(img_t)
        end_time = time.perf_counter()

    probs = torch.softmax(output, dim=1)
    max_prob = probs.max().item()
    pred = torch.argmax(output, 1).item()
    return class_names[pred], max_prob, end_time - start_time  # Trả về class, confidence, và inference time

if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"
    if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            prediction, confidence, inference_time = predict_image(image)
            print("Predicted class:", prediction, "Confidence:", confidence, "Inference time:", inference_time)
    else:
            print(f"File not found: {image_path}")