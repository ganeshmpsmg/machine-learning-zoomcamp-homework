import os
import shutil

src_root = os.path.abspath(".")
dest_dir = os.path.abspath("Midterm-Project")

print(f"Source: {src_root}")
print(f"Destination: {dest_dir}")

os.makedirs(dest_dir, exist_ok=True)

dirs_to_copy = ["data", "models", "tests", "scripts"]
files_to_copy = [
    "notebook.ipynb",
    "pipeline_utils.py",
    "train.py",
    "predict.py",
    "app.py",
    "requirements.txt",
    "Dockerfile",
    "docker-compose.yml",
    "benchmark_results.json",
    "README.md"
]

for d in dirs_to_copy:
    src_d = os.path.join(src_root, d)
    dest_d = os.path.join(dest_dir, d)
    if os.path.exists(src_d):
        if os.path.exists(dest_d):
            shutil.rmtree(dest_d)
        shutil.copytree(src_d, dest_d)
        print(f"Copied directory: {d} -> {dest_d}")

for f in files_to_copy:
    src_f = os.path.join(src_root, f)
    dest_f = os.path.join(dest_dir, f)
    if os.path.exists(src_f):
        shutil.copy2(src_f, dest_f)
        print(f"Copied file: {f} -> {dest_f}")

print("\nFiles in Midterm-Project:")
for item in os.listdir(dest_dir):
    print(" -", item)
