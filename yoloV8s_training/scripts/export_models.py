from ultralytics import YOLO
import os

# === Config ===
MODEL_PATH = 'runs/detect/benchmark_yolov8s_full/weights/best.pt' 
EXPORT_DIR = 'exports'  # Cartella di output (verrà creata se non esiste)
IMG_SIZE = 640

# Caricamento modello
model = YOLO(MODEL_PATH)

# Esportazione TorchScript
model.export(format='torchscript', imgsz=IMG_SIZE, simplify=True, optimize=False)


# Esportazione ONNX
model.export(format='onnx', imgsz=IMG_SIZE, dynamic=True, simplify=True, optimize=False)


# Controllo file
print("[INFO] File esportati:")
for ext in ['.onnx', '.torchscript']:
    for file in os.listdir(''):
        if file.endswith(ext):
            print(" -", file)