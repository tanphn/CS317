import torch
import os 
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report
import json
from clearml import Task
import neptune

def evaluate():
    # Khởi tạo ClearML task
    task = Task.init(project_name="pipelinedoanmonhoc", task_name="Evaluate")

    # Khởi tạo Neptune run
    run = neptune.init_run(
            project="acclonecolab/FER",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmZjE5YjUyYS1lMTg1LTQ2ZDMtODQ2NS1mMDYxZmRiZjhjZDgifQ=="
        )


    # Kiểm tra thiết bị (CPU hoặc GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Định nghĩa transform cho dữ liệu
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    
    # Đổi từ val sang test
    test_ds = datasets.ImageFolder("data/processed/test", transform=transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32)

    # Tải mô hình
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(test_ds.classes))
    model.load_state_dict(torch.load("models/checkpoints/best_model.pt"))
    model = model.eval().to(device)

    # Đánh giá mô hình
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1).cpu()
            y_pred.extend(preds.tolist())
            y_true.extend(y.tolist())

    # Tạo báo cáo phân loại
    report = classification_report(y_true, y_pred, output_dict=True)
    with open("metrics/eval.json", "w") as f:
        json.dump(report, f, indent=2)

    # Log metric vào Neptune
    run["eval/accuracy"] = report["accuracy"]
    run["eval/weighted_f1"] = report["weighted avg"]["f1-score"]
    for class_idx, metrics in report.items():
        if class_idx.isdigit():
            run[f"eval/class_{class_idx}/precision"] = metrics["precision"]
            run[f"eval/class_{class_idx}/recall"] = metrics["recall"]
            run[f"eval/class_{class_idx}/f1"] = metrics["f1-score"]

    # Log metric vào ClearML
    task.get_logger().report_scalar("eval", "accuracy", value=report["accuracy"], iteration=0)
    task.get_logger().report_scalar("eval", "weighted_f1", value=report["weighted avg"]["f1-score"], iteration=0)

    # Upload artifact vào ClearML
    task.upload_artifact("eval_report", artifact_object="metrics/eval.json")

    run.stop()

if __name__ == "__main__":
    evaluate()
