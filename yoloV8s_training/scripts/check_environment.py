import os
import sys
import importlib.util

# Config
MODEL_ARCH = 'yolov8s.pt'
DATASET = 'coco.yaml'
COCO_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'coco'))
REQUIRED_DIRS = ['train2017', 'val2017', 'annotations']
REQUIRED_PKGS = ['ultralytics', 'matplotlib', 'pandas']

DEVICE = 0  # 0 = GPU, 'cpu' = CPU

print("\n[CHECK] Controllo ambiente YOLOv8...\n")

# Python version
print(f"Python: {sys.executable}")
print(f"Versione: {sys.version.split()[0]}")

# pacchetti installati
for pkg in REQUIRED_PKGS:
    if importlib.util.find_spec(pkg) is None:
        print(f"Manca il pacchetto: {pkg}")
        sys.exit(1)
    else:
        print(f"Pacchetto presente: {pkg}")

# modello
if not os.path.isfile(MODEL_ARCH):
    print(f"Modello non trovato: {MODEL_ARCH}")
    sys.exit(1)
else:
    print(f"Modello YOLO trovato: {MODEL_ARCH}")

# file YAML
if not os.path.isfile(DATASET):
    print(f"File dataset YAML non trovato: {DATASET}")
    sys.exit(1)
else:
    print(f"File YAML trovato: {DATASET}")

# cartelle dataset
for folder in REQUIRED_DIRS:
    path = os.path.join(COCO_BASE, folder)
    if not os.path.isdir(path):
        print(f"Cartella mancante: {path}")
        sys.exit(1)
    else:
        print(f"Cartella trovata: {path}")

# GPU
try:
    import torch
    if DEVICE != 'cpu' and not torch.cuda.is_available():
        print("GPU non disponibile, ma richiesta. Imposta DEVICE='cpu' o installa PyTorch con CUDA.")
        sys.exit(1)
    elif DEVICE != 'cpu':
        print(f"GPU rilevata: {torch.cuda.get_device_name(DEVICE)}")
    else:
        print("Uso CPU abilitato.")
except ImportError:
    print("torch non installato.")
    sys.exit(1)

print("\nAmbiente OK. Puoi procedere con l'addestramento")
