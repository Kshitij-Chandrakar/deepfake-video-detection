# here we have combined dataset and renamed them to a common format for easier handling
import os
import shutil
import random

# ===== PATHS =====
source_path = r"E:\Project_Deepfake\Frames"

train_real = r"E:\Project_Deepfake\Frames\train\real"
train_fake = r"E:\Project_Deepfake\Frames\train\fake"
val_real   = r"E:\Project_Deepfake\Frames\val\real"
val_fake   = r"E:\Project_Deepfake\Frames\val\fake"

# Create folders
for path in [train_real, train_fake, val_real, val_fake]:
    os.makedirs(path, exist_ok=True)


# ===== FUNCTION =====
def split_data(source_folder, train_folder, val_folder, split_ratio=0.8):
    files = [f for f in os.listdir(source_folder) if f.endswith((".jpg", ".png"))]
    
    random.shuffle(files)
    
    split_index = int(len(files) * split_ratio)
    
    train_files = files[:split_index]
    val_files = files[split_index:]

    for file in train_files:
        shutil.copy(os.path.join(source_folder, file),
                    os.path.join(train_folder, file))

    for file in val_files:
        shutil.copy(os.path.join(source_folder, file),
                    os.path.join(val_folder, file))

    print(f"{source_folder} -> Train: {len(train_files)}, Val: {len(val_files)}")


# ===== RUN =====
print("Splitting REAL...")
split_data(
    os.path.join(source_path, "real"),
    train_real,
    val_real
)

print("Splitting FAKE...")
split_data(
    os.path.join(source_path, "fake"),
    train_fake,
    val_fake
)

print("Dataset split completed!")