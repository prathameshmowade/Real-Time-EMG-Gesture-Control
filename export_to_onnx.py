"""
======================================================================
  EMG MODEL ONNX EXPORTER
  Converts trained PyTorch models (TCN, BiLSTM, DTSF-CNN) to the universal
  ONNX standard (.onnx) for deployment in C++, C#, Java, JavaScript,
  WebAssembly, Android, and iOS without PyTorch!
======================================================================
"""

import os
import json
import torch
import numpy as np

def export_tcn_to_onnx(model_path="tcn_model.pth", output_path="tcn_model.onnx"):
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        return

    from train_tcn_fast import TemporalConvNet
    model = TemporalConvNet(n_classes=6)
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(model_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()

    # Dummy inputs: raw signal (1, 1, 256) and handcrafted features (1, 15)
    dummy_signal = torch.randn(1, 1, 256, dtype=torch.float32)
    dummy_features = torch.randn(1, 15, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_signal, dummy_features),
        output_path,
        input_names=["raw_signal", "handcrafted_features"],
        output_names=["gesture_logits"],
        dynamic_axes={
            "raw_signal": {0: "batch_size"},
            "handcrafted_features": {0: "batch_size"},
            "gesture_logits": {0: "batch_size"}
        },
        opset_version=18
    )
    print(f" Successfully exported TCN model to ONNX: {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")

if __name__ == "__main__":
    export_tcn_to_onnx()
