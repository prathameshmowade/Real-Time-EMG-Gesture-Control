"""
======================================================================
  PUBLIC REAL-WORLD EMG BENCHMARK DATASET LOADER & PREPROCESSOR
  Downloads and prepares open-access sEMG datasets (e.g. UCI EMG Gestures,
  NinaPro, EMG-EPN-612) for training models in this repository.
======================================================================
"""

import os
import sys
import json
import urllib.request
import zipfile
import numpy as np
import pandas as pd

PUBLIC_DATASETS = {
    "1": {
        "name": "UCI Machine Learning Repository - EMG Data for Gestures",
        "description": "36 subjects performing 6 basic hand/wrist gestures recorded using 2 surface EMG channels at 1000 Hz.",
        "subjects": 36,
        "channels": 2,
        "gestures": ["Hand at rest (Relax)", "Hand clenched in fist", "Wrist flexion", "Wrist extension", "Radial deviations", "Ulnar deviations"],
        "url": "https://archive.ics.uci.edu/static/public/481/emg+data+for+gestures.zip",
        "downloadable": True
    },
    "2": {
        "name": "NinaPro Database (DB1, DB2, DB5)",
        "description": "Gold standard benchmarking database for advanced robotic prosthetics control with up to 52 gestures across 67+ subjects.",
        "subjects": "27 (DB1), 40 (DB2), 10 (DB5)",
        "channels": "10 to 16 sEMG channels",
        "url": "http://ninapro.hevs.ch/",
        "downloadable": False,
        "access_note": "Requires free academic registration on the NinaPro portal."
    },
    "3": {
        "name": "EMG-EPN-612 Hand Gesture Dataset",
        "description": "612 trials across 30 subjects performing 5 hand gestures using the 8-channel Thalmic Myo Armband (200 Hz).",
        "subjects": 30,
        "channels": 8,
        "url": "https://ieee-dataport.org/open-access/emg-epn-612-dataset",
        "downloadable": False,
        "access_note": "Available on IEEE DataPort (Open Access)."
    },
    "4": {
        "name": "CapgMyo High-Density sEMG Dataset",
        "description": "High-density 128-electrode array recording 8 isometric hand gestures across 23 intact subjects.",
        "subjects": 23,
        "channels": 128,
        "url": "http://zju-capg.org/myo/data/",
        "downloadable": False,
        "access_note": "Available via Zhejiang University CAPG research group."
    }
}

def download_uci_dataset(dest_dir="dataset/uci_emg"):
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "emg_data_for_gestures.zip")
    url = PUBLIC_DATASETS["1"]["url"]

    print("=" * 65)
    print("   DOWNLOADING UCI EMG GESTURES DATASET")
    print("=" * 65)
    print(f" URL: {url}")
    print(f" Destination: {zip_path}")
    print(" Downloading (approx. ~15 MB)... Please wait.")

    try:
        urllib.request.urlretrieve(url, zip_path)
        print(" Download complete! Extracting archive...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        print(f" Extracted to: {dest_dir}")
        print(" Dataset is ready for training!")
    except Exception as e:
        print(f" Download error: {e}")
        print(f" You can manually download the zip file from:\n {url}\n and extract it to '{dest_dir}'.")

def display_dataset_catalog():
    print("=" * 70)
    print("   🌐 TOP REAL-WORLD EMG BENCHMARK DATASETS FOR GESTURE CONTROL")
    print("=" * 70)
    for key, ds in PUBLIC_DATASETS.items():
        print(f"\n[{key}] {ds['name']}")
        print(f"    • Description: {ds['description']}")
        print(f"    • Subjects: {ds['subjects']} | Channels: {ds['channels']}")
        print(f"    • Official URL: {ds['url']}")
        if "access_note" in ds:
            print(f"    • Access: {ds['access_note']}")
    print("=" * 70)

if __name__ == "__main__":
    display_dataset_catalog()
    print("\nTo automatically download the UCI EMG benchmark dataset:")
    print("  python download_public_datasets.py --download-uci")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--download-uci":
        download_uci_dataset()
