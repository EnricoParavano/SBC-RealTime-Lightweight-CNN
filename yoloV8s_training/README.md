

# YOLOv8s Edge Computing Project

Pipeline completa per training, export, benchmark e deployment di modelli YOLOv8s ottimizzati per edge computing.

## Panoramica

Questo progetto implementa un workflow end-to-end per:
- Addestrare YOLOv8s su dataset COCO
- Esportare modelli in formati edge-friendly (TorchScript, ONNX)
- Eseguire benchmark quantitativi su GPU e CPU
- Validare deployment su edge devices

Il progetto è stato sviluppato come parte di una tesi su **ottimizzazione di modelli CNN per edge computing**, con focus su YOLOv8s per object detection su dispositivi embedded.

---

## Struttura Repository

```
YOLOv8s-Edge-Project/
├── data/                      # Dataset COCO (da preparare)
├── models/                    # Modelli addestrati ed esportati
├── configs/                   # File YAML configurazione
│   ├── coco.yaml             # Config COCO2017 completo
│   └── coco8.yaml            # Config COCO8 ridotto (test)
├── scripts/                   # Pipeline principale
│   ├── check_environment.py  # Verifica setup ambiente
│   ├── train_yolov8s.py      # Training completo
│   ├── export_models.py      # Export TorchScript/ONNX
│   ├── benchmark_models.py   # Benchmark multi-formato
│   └── utils/                # Script di supporto
│       ├── check_annotations.py
│       ├── fix_coco_structure.py
│       └── onnx_cuda_test.py
├── runs/                      # Output training/validazione (auto-generato)
├── benchmarks.log            # Log risultati benchmark
└── README.md                 # Questa documentazione
```

---

## Requisiti

### Hardware
- **GPU**: NVIDIA con CUDA (consigliato RTX 3060 o superiore)
- **RAM**: Minimo 16 GB
- **Spazio disco**: 30+ GB (dataset COCO completo)

### Software
- Python >= 3.8
- CUDA >= 11.7 (per GPU)
- PyTorch >= 2.0 con CUDA support

### Librerie Python
```
ultralytics
torch
torchvision
matplotlib
pandas
onnxruntime-gpu  # o onnxruntime per solo CPU
```
---

## Dataset

### COCO8 (Test rapido)
- **Immagini**: 8 train + 8 val
- **Uso**: Test rapidi, debug, sviluppo iterativo
- **Download**: Automatico da Ultralytics al primo uso
- **Tempo training**: ~2 minuti (10 epoche)

### COCO2017 (Training completo)
- **Immagini**: ~118,000 train + ~5,000 val
- **Uso**: Training finale, benchmark pubblicabili
- **Download**: Manuale da [COCO Dataset](https://cocodataset.org/#download)
- **Spazio**: ~25 GB
- **Tempo training**: ~8-12 ore (100 epoche, RTX 3060)

**Struttura dataset attesa:**
```
data/COCO2017/
├── images/
│   ├── train2017/
│   └── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

---

## Setup Iniziale

### 1. Clone Repository
```
git clone https://github.com/EnricoParavano/YOLOv8s-Edge-Project.git
cd YOLOv8s-Edge-Project
```

### 2. Crea Virtual Environment
```
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure
venv\Scripts\activate     # Windows
```

### 3. Installa Dipendenze
**Installa Dipendenze**: Scarica tutte le dipenze necessarie citate prima

### 4. Download Dataset
**COCO8**: Scaricato automaticamente al primo uso

**COCO2017**: 
1. Download da [cocodataset.org](https://cocodataset.org/#download)
2. Estrai in `data/COCO2017/`
3. Verifica struttura cartelle corretta

---

## Piplen di Utilizzo

### Fase 1: Configurazione e Verifica

#### 1.1 Verifica Ambiente
```
python scripts/check_environment.py
```

**Cosa fa:**
- Controlla versione Python e librerie installate
- Verifica presenza o meno del file modello (`yolov8s.pt`) e dataset YAML
- Controlla struttura cartelle dataset
- Rileva GPU/CUDA disponibile

**Parametri configurabili** (modifica in cima allo script):
```
MODEL_ARCH = 'yolov8s.pt'
DATASET = 'coco.yaml'  # o 'coco8.yaml'
DEVICE = 0             # 0=GPU, 'cpu'=CPU
```

**Output successo:**
```
[CHECK] Controllo ambiente YOLOv8...
Python: /usr/bin/python3
Versione: 3.10.12
Pacchetto presente: ultralytics
...
GPU rilevata: NVIDIA GeForce RTX 3060
Ambiente OK. Puoi procedere con l'addestramento
```

---

#### 1.2 Verifica Annotazioni (Opzionale)

Test rapido (1 epoca) per validare dataset e annotazioni:

```
python scripts/utils/check_annotations.py
```

**Cosa fa:** Esegue 1 epoca di training con batch minimo per verificare correttezza dataset

**Tempo:** ~5-10 secondi (COCO8), ~2-5 minuti (COCO2017)

**Quando usarlo:**
- Dopo setup iniziale
- Prima di training lungo su COCO2017
- Dopo modifiche a file YAML

---

### Fase 2: Training

#### 2.1 Training YOLOv8s Completo

Script principale per addestramento con workflow automatizzato completo.

```
python scripts/train_yolov8s.py
```

**Cosa fa:**
1. Addestra YOLOv8s su dataset configurato
2. Salva checkpoint automatici (best.pt, last.pt)
3. Valida modello su validation set
4. Genera grafici metriche (loss, precision, recall, mAP)
5. Esegue benchmark finale

**Parametri configurabili** (modifica in cima allo script):
```
MODEL_ARCH = 'yolov8s.pt'   # Modello base pretrained
DATASET = 'coco.yaml'       # Dataset (coco.yaml o coco8.yaml)
EPOCHS = 100                # Numero epoche
BATCH = 64                  # Batch size
IMG_SIZE = 640              # Dimensione immagini
DEVICE = 0                  # 0=GPU, 'cpu'=CPU
PATIENCE = 10               # Early stopping patience
```

**Strategia consigliata (sviluppo iterativo):**

1. **Test rapido su COCO8**
   ```
   DATASET = 'coco8.yaml'
   EPOCHS = 10
   BATCH = 16
   ```
   Tempo: ~2 minuti

2. **Validazione intermedia**
   ```
   DATASET = 'coco8.yaml'
   EPOCHS = 50
   ```
   Tempo: ~10 minuti

3. **Training finale**
   ```
   DATASET = 'coco.yaml'
   EPOCHS = 100
   BATCH = 64
   ```
   Tempo: ~8-12 ore (RTX 3060)

**Output generato:**
```
runs/detect/train/
├── weights/
│   ├── best.pt          # ! Migliore ! modello (max mAP) - USA QUESTO
│   └── last.pt          # Ultimo checkpoint
├── results.csv          # Metriche per epoca
├── results.png          # Grafici training/validation
├── confusion_matrix.png
├── F1_curve.png
├── P_curve.png
├── R_curve.png
└── args.yaml            # Parametri usati
```

**Metriche visualizzate:**
- Loss (box, classification, DFL)
- Precision, Recall
- mAP@0.5, mAP@0.5:0.95

---

### Fase 3: Export Modelli

#### 3.1 Export per Deployment Edge

Dopo il training, esporta il modello in formati ottimizzati per deployment.

```
python scripts/export_models.py
```

**Cosa fa:**
- Carica `best.pt` dal training
- Converte in TorchScript e ONNX
- Ottimizza per inferenza edge

**Parametri configurabili:**
```
MODEL_PATH = 'runs/detect/train/weights/best.pt'
IMG_SIZE = 640
```

**Formati generati:**

| Formato | File | Dimensione | Uso |
|---------|------|------------|-----|
| TorchScript | `best.torchscript` | ~12 MB | Deploy PyTorch production |
| ONNX | `best.onnx` | ~25 MB | ONNX Runtime, TensorRT, OpenVINO |

**Output:**
```
[INFO] Caricamento modello da runs/detect/train/weights/best.pt...
TorchScript export success, saved as best.torchscript (12.5 MB)
ONNX export success, saved as best.onnx (24.8 MB)

[INFO] File esportati:
 - best.onnx
 - best.torchscript
```

**Opzioni export:**
- `simplify=True`: Semplifica grafo computazionale
- `dynamic=True` (ONNX): Batch size variabile per flessibilità
- `optimize=False`: Mantieni compatibilità massima

**Tempo esecuzione:** ~30 secondi per entrambi i formati

---

### Fase 4: Benchmark e Validazione

#### 4.1 Benchmark Multi-Formato

Confronta performance di tutti i formati esportati su GPU e CPU.

```
python scripts/benchmark_models.py
```

**Cosa fa:**
1. Valida PyTorch, TorchScript, ONNX su validation set
2. Testa ogni formato su GPU e CPU
3. Misura accuratezza e velocità
4. Genera tabella comparativa completa

**Parametri configurabili:**
```
DATASET = 'coco.yaml'       
IMG_SIZE = 640
DEVICES = [0, 'cpu']        # GPU e CPU

MODEL_PATHS = {
    "PyTorch": "runs/detect/train/weights/best.pt",
    "TorchScript": "runs/detect/train/weights/best.torchscript",
    "ONNX": "runs/detect/train/weights/best.onnx"
}
```

**Metriche raccolte:**
- **Accuratezza**: Precision, Recall, mAP@0.5, mAP@0.5:0.95
- **Performance**: Inference time (ms/immagine), FPS
- **Dimensione**: File size (MB)

**Output esempio:**
```
[INFO] Test su dispositivo: GPU
[INFO] Valutazione modello PyTorch su device 0...
[INFO] Valutazione modello TorchScript su device 0...
[INFO] Valutazione modello ONNX su device 0...

[INFO] Test su dispositivo: CPU
...

Benchmark completato:

   Formato Device Size (MB) Precision Recall  mAP50 mAP50-95 Inference (ms)   FPS
  PyTorch    GPU       24.5    0.8234 0.7912 0.8912   0.7123           12.8 78.13
TorchScript    GPU       12.3    0.8234 0.7912 0.8912   0.7123           11.5 86.96
      ONNX    GPU       24.8    0.8234 0.7912 0.8912   0.7123           10.2 98.04
  PyTorch    CPU       24.5    0.8234 0.7912 0.8912   0.7123          156.3  6.40
TorchScript    CPU       12.3    0.8234 0.7912 0.8912   0.7123          148.2  6.75
      ONNX    CPU       24.8    0.8234 0.7912 0.8912   0.7123          132.5  7.55
```

**Interpretazione risultati:**
- **Accuratezza**: Dovrebbe essere identica tra formati (differenze < 0.1% indicano problemi)
- **Velocità GPU**: ONNX tipicamente più veloce (ottimizzazioni ONNX Runtime)
- **Velocità CPU**: ONNX ottimizzato per Intel processors
- **Dimensione**: TorchScript più compatto (~50% PyTorch)

**Tempo esecuzione:**
- COCO8: ~1 minuto totale
- COCO2017: ~8 minuti totale (6 test × ~45 sec)

---

## Script di Supporto (Utils)

### check_annotations.py
Test rapido 1 epoca per validare dataset prima di training lungo.
```
python scripts/utils/check_annotations.py
```


### onnx_cuda_test.py
Test diagnostico veloce (~5 sec) per verificare compatibilità ONNX Runtime CUDA.
```
python scripts/utils/onnx_cuda_test.py
```

**Output successo:**
```
Provider disponibili: ['CUDAExecutionProvider', 'CPUExecutionProvider']
ONNX Runtime ha caricato correttamente il modello con CUDA!
```

**Fix problema CUDA:**
```
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

### GPU non rilevata
```
# Verifica driver NVIDIA
nvidia-smi

# Verifica PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstalla PyTorch con CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Risultati Attesi

### Metriche Training (COCO2017, 100 epoche)
- **Precision**: ~0.82
- **Recall**: ~0.79
- **mAP@0.5**: ~0.89
- **mAP@0.5:0.95**: ~0.71
