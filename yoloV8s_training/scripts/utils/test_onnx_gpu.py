from ultralytics import YOLO
import onnxruntime as ort
import pandas as pd

# === CONFIG ===
DATASET = 'coco.yaml'  # o il tuo file .yaml personalizzato
IMG_SIZE = 640
MODEL_PATH = 'runs/detect/benchmark_yolov8s_full/weights/best.onnx'

def check_onnx_gpu():
    try:
        sess = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider'])
        providers = sess.get_providers()
        print("[INFO] ONNX Runtime sta usando:", providers)
    except Exception as e:
        print("[ERRORE] ONNX Runtime non riesce a usare CUDA:", e)

def benchmark_onnx_model():
    print(f"[INFO] Avvio benchmark ONNX (GPU) su {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    metrics = model.val(data=DATASET, imgsz=IMG_SIZE, device=0, verbose=False)

    results = {
        "Formato": "ONNX (GPU)",
        "Precision": metrics.box.mp,
        "Recall": metrics.box.mr,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "Inference time (ms/im)": metrics.speed['inference'],
        "FPS": 1000.0 / metrics.speed['inference'] if metrics.speed['inference'] > 0 else 0
    }

    print("\n[RESULTATI]")
    print(pd.DataFrame([results]).to_string(index=False))

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    check_onnx_gpu()
    benchmark_onnx_model()
