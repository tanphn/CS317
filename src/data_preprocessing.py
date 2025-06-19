import os
import shutil
from pathlib import Path
import json
import random

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def preprocess():
    if (PROCESSED_DIR / "train").exists():
        print("Processed data already exists. Skipping preprocessing.")
        return

    # Xóa thư mục cũ nếu có
    shutil.rmtree(PROCESSED_DIR, ignore_errors=True)

    # Tạo các thư mục mới
    for split in ["train", "val", "test"]:
        os.makedirs(PROCESSED_DIR / split, exist_ok=True)

    # Chia dữ liệu 70% train, 20% val, 10% test
    for class_dir in RAW_DIR.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg"))
            random.shuffle(images)  # Trộn ngẫu nhiên

            n = len(images)
            train_end = int(0.7 * n)
            val_end = train_end + int(0.2 * n)

            train_imgs = images[:train_end]
            val_imgs = images[train_end:val_end]
            test_imgs = images[val_end:]

            for split, img_list in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
                split_class_dir = PROCESSED_DIR / split / class_dir.name
                os.makedirs(split_class_dir, exist_ok=True)
                for img in img_list:
                    shutil.copy(img, split_class_dir / img.name)

    # Ghi log JSON
    with open("metrics/preprocessing.json", "w") as f:
        json.dump({"status": "done"}, f)

if __name__ == "__main__":
    preprocess()
