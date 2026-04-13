from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import tensorflow as tf
import tempfile
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model2_tf")

model = tf.keras.models.load_model(MODEL_PATH)
# -------- Preprocess frame --------
def preprocess(frame):
    frame = cv2.resize(frame, (128, 128))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# -------- Extract frames --------
def extract_frames(video_path, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_skip == 0:
            frames.append(frame)

        count += 1

    cap.release()
    return frames

# -------- Prediction logic --------
def predict_video(frames):
    if len(frames) == 0:
        return "ERROR", 0.0

    processed_frames = []

    for frame in frames:
        frame = cv2.resize(frame, (128, 128))
        frame = frame / 255.0
        processed_frames.append(frame)

    processed_frames = np.array(processed_frames)

    # 🔥 batch prediction (IMPORTANT)
    preds = model.predict(processed_frames, verbose=0).flatten()

    print("Pred sample:", preds[:10])
    print("Min:", preds.min(), "Max:", preds.max())

    avg_pred = np.median(preds)

    label = "fake" if avg_pred > 0.5 else "real"

    return label, float(avg_pred)

# -------- API --------
@app.post("/predict-video/")
async def predict_video_api(file: UploadFile = File(...)):

    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        temp.write(await file.read())
        temp_path = temp.name

    # Extract frames
    frames = extract_frames(temp_path, frame_skip=10)

    # Predict
    label, confidence = predict_video(frames)

    # Cleanup
    os.remove(temp_path)

    return {
        "prediction": label,
        "confidence": confidence,
        "frames_used": len(frames)
    }