# YOLOv8s Training and Benchmark and Export models

Pipeline completa per training, export, benchmark e deployment di modelli YOLOv8s.

---

## Panoramica

Questo progetto implementa un workflow end-to-end per:
- Addestrare YOLOv8s su dataset COCO
- Esportare modelli in formati edge-friendly (TorchScript, ONNX, MNN, NCNN)
- Eseguire benchmark quantitativi su GPU e CPU
- Validare deployment su edge devices
- Integrare quantizzazione post-training (PTQ) per SBC

Il progetto è stato sviluppato come parte di una tesi su **ottimizzazione di modelli CNN per edge computing**, con focus su YOLOv8s per object detection su dispositivi embedded.

---

## 📁 Struttura Repository

```bash
YOLOv8s-Edge-Project/
├── data/                      # Dataset COCO (da scaricare da sito originale)
├── models/                    # Modelli addestrati ed esportati
├── configs/                   # File YAML configurazione
│   ├── coco.yaml              # Config COCO2017 completo
│   └── coco8.yaml             # Config COCO8 ridotto per test
├── scripts/                   # Pipeline principale
│   ├── check_environment.py   # Verifica setup ambiente
│   ├── train_yolov8s.py       # Training completo
│   ├── export_models.py       # Export TorchScript/ONNX
│   ├── benchmark_models.py    # Benchmark multi-formato
│   └── utils/                 # Script di supporto
│       ├── check_annotations.py
│       ├── fix_coco_structure.py
│       └── onnx_cuda_test.py
├── runs/                      # Output training/validazione (auto-generato)
├── benchmarks.log             # Log risultati benchmark
├── requirements.txt           # Dipendenze Python
└── README.md                  # Questa documentazione
```

---

## Requisiti

**Hardware:**
- GPU: NVIDIA RTX 3090 (nei test di riferimento; pipeline scalabile anche su GPU come RTX 3060)
- RAM: Minimo 16 GB
- Spazio disco: 30+ GB (dataset COCO completo)

**Software:**
- Python >= 3.11.9
- CUDA >= 11.8 (per GPU)
- PyTorch >= 2.6 con CUDA support

**Librerie Python:**
```bash
ultralytics
torch
torchvision
matplotlib
pandas
onnxruntime-gpu   # o onnxruntime per solo CPU
```
---

## 📊 Dataset

### COCO8 (Test rapido)
- Immagini: 8 train + 8 val  
- Uso: Test rapidi, debug, sviluppo iterativo  
- Download: Automatico da Ultralytics al primo uso  

### COCO2017 (Training completo)
- Immagini: ~118,000 train + ~5,000 val  
- Uso: Training finale, benchmark pubblicabili  
- Download: Manuale da [cocodataset.org](https://cocodataset.org/#download)  
- Spazio richiesto: ~25 GB  

**Struttura attesa:**
```bash
data/coco/
├── images/
│   ├── train2017/
│   └── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

---

## Setup Iniziale

1. **Clone Repository**
```bash
git clone https://github.com/EnricoParavano/YOLOv8s-Edge-Project.git
cd YOLOv8s-Edge-Project
```

2. **Crea Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scriptsctivate     # Windows
```

3. **Download Dataset**
- COCO8 → automatico al primo uso  
- COCO2017 → scarica e estrai in `data/coco/`


---

##  Pipeline di Utilizzo

###  Fase 1: Configurazione e Verifica

**1.1 Verifica Ambiente**
```bash
python scripts/check_environment.py
```

**1.2 Verifica Annotazioni (opzionale)**
```bash
python scripts/utils/check_annotations.py
```

---

###  Fase 2: Training

**Comando:**
```bash
python scripts/train_yolov8s.py
```

**Parametri principali:**
```python
MODEL_ARCH = 'yolov8s.pt'
DATASET = 'coco.yaml'     # o 'coco8.yaml'
EPOCHS = 100
BATCH = 64
IMG_SIZE = 640
DEVICE = 0
PATIENCE = 10
```

---

###  Fase 3: Export Modelli

**Comando:**
```bash
python scripts/export_models.py
```

| Formato      | File              | Uso                              |
|--------------|------------------|----------------------------------|
| TorchScript  | best.torchscript | Deploy PyTorch production        |
| ONNX         | best.onnx        | ONNX Runtime, TensorRT, OpenVINO |
| MNN          | best.mnn         | Ottimizzato per mobile/SBC       |
| NCNN         | best.ncnn        | Ottimizzato per mobile/SBC       |

---

###  Fase 4: Benchmark e Validazione

**Comando:**
```bash
python scripts/benchmark_models.py
```

**Metriche raccolte:**
- Precision, Recall, mAP@0.5, mAP@0.5:0.95  
- Inference time, FPS  
- Dimensione modello (MB)  

---

##  Quantizzazione (PTQ)

Dopo l’addestramento completo, il modello può essere quantizzato in INT8 per l’esecuzione su SBC (Raspberry Pi, Jetson Orin, ecc.).

Esempio (PyTorch → TFLite INT8):

```bash
python export.py --weights runs/detect/train/weights/best.pt --include tflite --int8
```

Questa fase riduce la dimensione del modello e accelera l’inferenza su CPU a basso consumo, con un lieve calo di accuratezza.

---

##  Script di Supporto (Utils)

- `check_annotations.py` → Validazione dataset rapida  
- `fix_coco_structure.py` → Correzione struttura cartelle dataset  
- `onnx_cuda_test.py` → Test compatibilità CUDA in ONNX Runtime  

---

##  Troubleshooting

- **Bugfix YOLOv8 (Aprile 2025)**: sostituire `metrics.box.map50to95` con `metrics.box.map()` per calcolo mAP corretto.

- **GPU non rilevata**
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

- **Out of Memory (OOM)** → Riduci batch size  
- **Cartelle dataset errate** → `fix_coco_structure.py`  
- **ONNX Runtime senza CUDA**  
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

---

##  Riferimenti

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- [COCO Dataset](https://cocodataset.org/)  
- [ONNX Runtime](https://onnxruntime.ai/)  
- [PyTorch](https://pytorch.org/)  

---

##  Citazione

```bibtex
@misc{yolov8s-edge-project,
  author = {Enrico Paravano},
  title = {YOLOv8s Edge Computing Project},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/EnricoParavano/YOLOv8s-Edge-Project}
}
```

---

## 👤 Autore

**Enrico Paravano**  
Tesi: *Ottimizzazione modelli CNN per edge computing*

---

## 📄 Licenza

Questo progetto utilizza [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), rilasciato sotto licenza AGPL-3.0.

---

## 📊 Appendice — Esempio Sessione Benchmark (RTX 3090)

| Formato     | mAP50-95 | Inference Time | FPS   |
|-------------|----------|----------------|-------|
| PyTorch     | 0.4111   | 11.95 ms       | 83.69 |
| TorchScript | 0.4089   | 8.53 ms        | 117.2 |
| ONNX        | 0.4089   | 10.84 ms       | 92.22 |
| MNN         | 0.4089   | 61.21 ms       | 16.34 |
| NCNN        | 0.4089   | 96.87 ms       | 10.32 |

Valori ottenuti con: **YOLOv8s, batch=16, 1 epoca, input 640x640, GPU RTX 3090**.
