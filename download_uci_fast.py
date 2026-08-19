"""
Download and extract UCI / Kaggle EMG Data for Gestures (36 subjects, 2 channels)
"""
import os
import sys
import zipfile
import urllib.request
import numpy as np
import pandas as pd

def download_and_extract_uci():
    dest_dir = "dataset/uci_emg"
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "emg_data_for_gestures.zip")
    url = "https://archive.ics.uci.edu/static/public/481/emg+data+for+gestures.zip"

    print("=" * 65)
    print("   DOWNLOADING REAL-WORLD UCI / KAGGLE EMG DATASET")
    print("=" * 65)
    print(f" URL: {url}")
    print(f" Target: {zip_path}")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=30) as response, open(zip_path, "wb") as out_file:
        downloaded = 0
        chunk_size = 1024 * 512  # 512 KB
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            print(f"\r Downloaded: {downloaded / (1024*1024):.2f} MB...", end="", flush=True)

    print(f"\n Download Complete! Total Size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB")
    print(" Extracting archive...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_dir)
    print(f" Successfully extracted to: {dest_dir}")

if __name__ == "__main__":
    download_and_extract_uci()
