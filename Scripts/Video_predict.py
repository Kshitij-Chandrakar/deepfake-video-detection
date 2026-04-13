import cv2
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model(r"E:\Project_Deepfake\Models\deepfake_model1.keras")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def predict_video_face(video_path):
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 5 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5
            )

            if len(faces) == 0:
                continue

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]

                face = cv2.resize(face, (128, 128))
                face = face / 255.0
                face = np.expand_dims(face, axis=0)

                pred = model.predict(face, verbose=0)[0][0]
                predictions.append(pred)

                break

            if len(predictions) >= 10:
                break

        frame_count += 1

    cap.release()

    if len(predictions) == 0:
        print("No face detected ❌")
        return

    avg_pred = np.mean(predictions)

    print(f"Average prediction: {avg_pred:.4f}")

    if avg_pred > 0.5:
        print("Result: REAL")
    else:
        print("Result: FAKE")


# ===== RUN =====
predict_video_face(r"C:\Users\chand\Videos\Recording 2026-04-09 224023.mp4")