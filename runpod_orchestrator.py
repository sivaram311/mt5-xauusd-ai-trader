"""
Package and deploy Mean Reversion bundle to RunPod, train model, download ONNX, and terminate pod.
"""
import os
import sys
import time
import zipfile
import json
import urllib.request
import urllib.error

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
ZIP_PATH = r"W:\data\trading\runpod_mean_reversion_bundle.zip"

def create_bundle():
    print("1. Packaging RunPod training bundle...")
    files_to_pack = [
        (r"W:\data\trading\xauusd_m5_mean_reversion_dataset.npz", "xauusd_m5_mean_reversion_dataset.npz"),
        (r"W:\data\trading\train_runpod_mean_reversion.py", "train_runpod_mean_reversion.py"),
    ]
    with zipfile.ZipFile(ZIP_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        for src, arc in files_to_pack:
            if os.path.exists(src):
                zipf.write(src, arcname=arc)
                print(f"   Added {arc} ({os.path.getsize(src)/(1024*1024):.2f} MB)")
            else:
                print(f"   Error: File {src} not found!")
                return False
    print(f"Bundle ready: {ZIP_PATH} ({os.path.getsize(ZIP_PATH)/(1024*1024):.2f} MB)")
    return True

if __name__ == "__main__":
    if create_bundle():
        print("Dataset and training code bundle successfully prepared!")
