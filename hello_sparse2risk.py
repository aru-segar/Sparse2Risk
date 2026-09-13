import sys
from pathlib import Path

print("Sparse2Risk environment is working.")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

dataset_root = Path(r"C:\FYP\dataset")

print(f"KITTI dataset path: {dataset_root}")
print(f"Dataset exists: {dataset_root.exists()}")