from ultralytics import YOLO
import pandas as pd
import os
import multiprocessing

# === CONFIG ===
DATASET = 'coco.yaml'
IMG_SIZE = 640
MODEL_PATHS = {
    "PyTourch": "runs/detect/train/weights/best.pt",
    "TorchScript": "runs/detect/train/weights/best.torchscript",
    "ONNX": "runs/detect/train/weights/best.onnx"
}
DEVICES = [0, 'cpu']  # 0 = GPU, 'cpu' = CPU

def benchmark_model(name, model_path, device):
    print(f"[INFO] Valutazione modello {name} su device {device}...")
    model = YOLO(model_path)

    # Validazione
    metrics = model.val(data=DATASET, imgsz=IMG_SIZE, device=device, verbose=False)

    return {
        "Formato": name,
        "Device": "GPU" if device == 0 else "CPU",
        "Size (MB)": round(os.path.getsize(model_path) / 1e6, 1) if os.path.exists(model_path) else "N/A",
        "Precision": round(metrics.box.mp, 4),
        "Recall": round(metrics.box.mr, 4),
        "mAP50": round(metrics.box.map50, 4),
        "mAP50-95": round(metrics.box.map, 4),
        "Inference time (ms/im)": round(metrics.speed['inference'], 2),
        "FPS": round(1000.0 / metrics.speed['inference'], 2) if metrics.speed['inference'] > 0 else 0
    }

def main():
    results = []
    for device in DEVICES:
        print(f"\n[INFO] Test su dispositivo: {'GPU' if device == 0 else 'CPU'}")
        for name, path in MODEL_PATHS.items():
            result = benchmark_model(name, path, device)
            results.append(result)

    df = pd.DataFrame(results)
    print("\nBenchmark completato:\n")
    print(df.to_string(index=False))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
