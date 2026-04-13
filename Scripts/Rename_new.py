import os

# ===== BASE PATH =====
base_path = r"E:\Project Deepfake\videos"

# ===== FOLDERS TO PROCESS =====
folders = [
    ("train/Real", "real"),
    ("train/Fake", "fake"),
    ("val/Real", "real"),
    ("val/Fake", "fake"),
]

for folder, prefix in folders:
    folder_path = os.path.join(base_path, folder)
    
    files = sorted(os.listdir(folder_path))
    
    for i, file in enumerate(files):
        if not file.endswith(".mp4"):
            continue
        
        new_name = f"{prefix}_{i:03d}.mp4"
        
        old_path = os.path.join(folder_path, file)
        new_path = os.path.join(folder_path, new_name)
        
        os.rename(old_path, new_path)
    
    print(f"Renamed files in: {folder}")

print("All folders renamed successfully!")