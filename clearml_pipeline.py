from clearml import PipelineDecorator, Task
from clearml import Task

Task.set_credentials(
    api_host="https://api.clear.ml",
    web_host="https://app.clear.ml",
    files_host="https://files.clear.ml",
    key="PFU8WHA02X3W7ZXD1X2JC1M0BTQN87",
    secret="pQ00sMRzTAuJK34fcjpHLDUWEw070ip8aIJ14Frpi5ud5MoKEQ0Vj7xnx937SWBmN8I"
)
# Định nghĩa các bước (steps) trong pipeline
@PipelineDecorator.component(return_values=["status"], execution_queue="default")
def preprocess_step():
    print("Running preprocessing step...")
    # Gọi script hoặc logic preprocessing
    import subprocess
    subprocess.run(["python", "./src/data_preprocessing.py"], check=True)
    return "preprocess_done"

@PipelineDecorator.component(return_values=["status"], execution_queue="default")
def train_step(preprocess_status):
    print(f"Preprocessing status: {preprocess_status}")
    print("Running training step...")
    # Gọi script hoặc logic training
    import subprocess
    subprocess.run(["python", "./src/train.py"], check=True)
    return "train_done"

@PipelineDecorator.component(return_values=["status"], execution_queue="default")
def evaluate_step(train_status):
    print(f"Training status: {train_status}")
    print("Running evaluation step...")
    # Gọi script hoặc logic evaluation
    import subprocess
    subprocess.run(["python", "./src/evaluate.py"], check=True)
    return "evaluate_done"

# # Định nghĩa pipeline chính
# @PipelineDecorator.pipeline(
#     name="pipelinedoanmonhoc",
#     project="Doan",
#     version="1.0"
# )
@PipelineDecorator.pipeline(
    name="pipelinedoanmonhoc",
    project="Doan",
    version="1.0",
    pipeline_execution_queue="default"  # 👈 THÊM DÒNG NÀY
)
def run_pipeline():
    # Gọi các bước theo thứ tự
    preprocess_status = preprocess_step()
    train_status = train_step(preprocess_status)
    evaluate_status = evaluate_step(train_status)
    print(f"Pipeline completed with evaluation status: {evaluate_status}")

# Chạy pipeline
# if __name__ == "__main__":
#     # Đảm bảo các bước được chạy với mã hóa UTF-8
#     import os
#     os.environ["PYTHONIOENCODING"] = "utf-8"
    
#     # Bật PipelineDecorator để chạy pipeline
#     PipelineDecorator.run_locally()  # Chạy locally
#     run_pipeline()

if __name__ == "__main__":
    import os
    os.environ["PYTHONIOENCODING"] = "utf-8"
    
    # Tạo một Task controller để gửi pipeline lên server
    task = Task.init(
        project_name="Doan",
        task_name="Pipeline Run (Agent)",
        task_type=Task.TaskTypes.controller
    )
    
    # Không cần run_locally nữa
    run_pipeline()
