from fastapi import FastAPI, UploadFile, File
from prometheus_fastapi_instrumentator import Instrumentator
from predictor import predict_image
from PIL import Image
import io
import logging
from fastapi.responses import JSONResponse
from prometheus_client import Summary, Gauge

app = FastAPI()

# Cấu hình logging
logging.basicConfig(filename='/app/app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Định nghĩa metric tùy chỉnh
inference_time_summary = Summary('model_inference_time_seconds', 'Time spent in model inference', ['job'])
confidence_score = Gauge('model_confidence_score', 'Model confidence score')

# Thêm Prometheus instrumentation
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logging.info("Received request for prediction")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Dự đoán và lấy thời gian suy luận
        pred_class, confidence, inference_time = predict_image(image)
        
        # Ghi metric
        inference_time_summary.labels(job="fastapi").observe(inference_time)
        confidence_score.set(confidence)
        
        logging.info(f"Predicted class: {pred_class}, Inference time: {inference_time}s, Confidence: {confidence}")
        return {"predicted_class": pred_class, "inference_time": inference_time, "confidence": confidence}
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})