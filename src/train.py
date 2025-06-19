import os
import torch
import os 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score
import optuna
import neptune
from clearml import Task

def train_model(trial):
    # Khởi tạo Neptune run
    run = neptune.init_run(
            project="acclonecolab/FER",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmZjE5YjUyYS1lMTg1LTQ2ZDMtODQ2NS1mMDYxZmRiZjhjZDgifQ=="
        )



    # Khởi tạo ClearML task
    task = Task.init(project_name="pipelinedoanmonhoc", task_name="Training + Tuning")
    task.connect_configuration("config/clearml.conf")

    # Kiểm tra thiết bị (CPU hoặc GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Log hyperparameters vào Neptune
    run["parameters/lr"] = lr
    run["parameters/batch_size"] = batch_size

    # Định nghĩa transform cho dữ liệu
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_ds = datasets.ImageFolder("data/processed/train", transform=transform)
    val_ds = datasets.ImageFolder("data/processed/val", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Tải mô hình ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Huấn luyện mô hình với 10 epoch
    for epoch in range(10):  # Tăng từ 3 lên 10 epoch
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/10, Train Loss: {avg_train_loss}")
        run["train/loss"].log(avg_train_loss)  # Log train loss

        # Đánh giá trên tập train (tính train accuracy)
        model.eval()
        all_train_preds, all_train_labels = [], []
        with torch.no_grad():
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                preds = model(x).argmax(dim=1).cpu()
                all_train_preds.extend(preds.tolist())
                all_train_labels.extend(y.cpu().tolist())

        train_acc = accuracy_score(all_train_labels, all_train_preds)
        print(f"Epoch {epoch+1}/10, Train Accuracy: {train_acc}")
        run["train/accuracy"].log(train_acc)  # Log train accuracy

        # Đánh giá trên tập validation
        all_val_preds, all_val_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1).cpu()
                all_val_preds.extend(preds.tolist())
                all_val_labels.extend(y.cpu().tolist())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_val_labels, all_val_preds)
        print(f"Epoch {epoch+1}/10, Val Loss: {avg_val_loss}, Val Accuracy: {val_acc}")
        run["val/loss"].log(avg_val_loss)  # Log val loss
        run["val/accuracy"].log(val_acc)  # Log val accuracy
    
    # Lưu mô hình
    checkpoint_dir = "models/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    run["model/checkpoint"].upload(checkpoint_path)

    run.stop()
    return val_acc  # Trả về validation accuracy để Optuna tối ưu

if __name__ == "__main__":
    try:
        study = optuna.create_study(direction="maximize")
        study.optimize(train_model, n_trials=5)

        # Lấy trial tốt nhất
        best_trial = study.best_trial
        print(f"Best trial: {best_trial.params}, Accuracy: {best_trial.value}")

        # Log trial tốt nhất vào Neptune
        run = neptune.init_run(
                project="acclonecolab/FER",
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmZjE5YjUyYS1lMTg1LTQ2ZDMtODQ2NS1mMDYxZmRiZjhjZDgifQ=="
            )


        run["best/parameters"] = best_trial.params
        run["best/accuracy"] = best_trial.value
        run.stop()

    except Exception as e:
        print(f"Error during training: {e}")
        raise
