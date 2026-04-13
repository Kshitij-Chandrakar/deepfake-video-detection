import os
import cv2

# ===== SETTINGS =====
FRAME_SKIP = 4
MAX_FRAMES = 3
IMG_SIZE = 160

# ===== PATHS =====
REAL_VIDEO_PATH = "E:/Project_Deepfake/videos/real"
FAKE_VIDEO_PATH = "E:/Project_Deepfake/videos/fake"

REAL_OUTPUT_PATH = "E:/Project_Deepfake/Frames/real"
FAKE_OUTPUT_PATH = "E:/Project_Deepfake/Frames/fake"

# ===== CREATE OUTPUT FOLDERS (if not exist) =====
os.makedirs(REAL_OUTPUT_PATH, exist_ok=True)
os.makedirs(FAKE_OUTPUT_PATH, exist_ok=True)

# ===== LOAD FACE DETECTOR =====
face_cascade = cv2.CascadeClassifier(
    "E:/Project_Deepfake/Scripts/haarcascade_frontalface_default.xml"
)

print(face_cascade.empty())  # should print False

# ===== FUNCTION =====
def extract_frames(video_folder, output_folder):
    for video_file in os.listdir(video_folder):

        if not video_file.endswith(".mp4"):
            continue

        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % FRAME_SKIP == 0:

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=6
                )

                if len(faces) == 0:
                    frame_count += 1
                    continue

                # take first face
                (x, y, w, h) = faces[0]
                face = frame[y:y+h, x:x+w]

                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

                filename = f"{video_file[:-4]}_{saved_count}.jpg"
                save_path = os.path.join(output_folder, filename)

                cv2.imwrite(save_path, face)
                saved_count += 1

                if saved_count >= MAX_FRAMES:
                    break

            frame_count += 1

        cap.release()
        print(f"Processed: {video_file}")

# ===== RUN =====
print("Extracting REAL videos...")
extract_frames(REAL_VIDEO_PATH, REAL_OUTPUT_PATH)

print("Extracting FAKE videos...")
extract_frames(FAKE_VIDEO_PATH, FAKE_OUTPUT_PATH)

print("Done")