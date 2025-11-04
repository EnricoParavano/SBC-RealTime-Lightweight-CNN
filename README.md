# SBC RealTime Lightweight CNNs

Repository unico per l'addestramento, l'ottimizzazione e il deployment realtime di reti neurali leggere per la detection su dispositivi embedded.

## 📚 Indice
1. [Struttura generale](#-struttura-generale)
2. [Prerequisiti](#-prerequisiti)
3. [YOLOv8s – Pipeline di training](#-yolov8s--pipeline-di-training)
4. [YOLOv8s – Post-Training Quantization (PTQ)](#-yolov8s--post-training-quantization-ptq)
5. [EfficientDet INT8 – Benchmark](#-efficientdet-int8--benchmark)
6. [Deployment su Raspberry Pi / SBC](#-deployment-su-raspberry-pi--sbc)
7. [Streaming UDP a bassa latenza](#-streaming-udp-a-bassa-latenza)
8. [Dataset e gestione dei dati](#-dataset-e-gestione-dei-dati)
9. [Riferimenti](#-riferimenti)

---

## 📦 Struttura generale
```
SBC-RealTime-Lightweight-CNNs/
├── data/                        # Segnaposto per dataset locali (COCO, campioni di calibrazione, ecc.)
├── efficientdet_tests/          # Script e report per EfficientDet INT8
├── raspberry_deployment/        # Deployment YOLOv8s TFLite ottimizzato per Raspberry Pi
├── streaming_UDP/               # Sender/receiver per streaming UDP a bassa latenza
├── yoloV8s_PTQ/                 # Output della quantizzazione INT8 di YOLOv8s
└── yoloV8s_training/            # Pipeline completa di training/export/benchmark YOLOv8s

```
Ogni macro-cartella dispone di script e README dedicati con istruzioni operative dettagliate.

---

## YOLOv8s – Pipeline di training
Directory: [`yoloV8s_training/`](yoloV8s_training/)  
Documentazione completa: [`yoloV8s_training/README.md`](yoloV8s_training/README.md)

### Obiettivi
- Addestrare YOLOv8s su COCO2017 (full o subset `coco8`).
- Automatizzare export in TorchScript e ONNX.
- Confrontare prestazioni CPU/GPU sui diversi formati.

### Contenuti principali
- `scripts/check_environment.py`: verifica librerie, GPU e struttura dataset.
- `scripts/train_yolov8s.py`: workflow end-to-end di training con early stopping, logging e grafici (`runs/detect/train`).
- `scripts/export_models.py`: converte `best.pt` in TorchScript/ONNX con opzioni di ottimizzazione.
- `scripts/benchmark_models.py`: benchmark multipiattaforma (precision/recall, mAP, FPS) su PyTorch, TorchScript e ONNX.
- `scripts/utils/`: strumenti di diagnostica e file utilizzati durante il tirocinio (controllo annotazioni, fix struttura COCO, test ONNX CUDA).
- `configs/coco.yaml` e `configs/coco8.yaml`: configurazioni dataset.

### Flusso consigliato
1. **Setup**: crea e attiva un virtualenv, installa Ultralytics e dipendenze.
2. **Verifica**: `python scripts/check_environment.py` per assicurarsi che dataset e GPU siano pronti.
3. **Training iterativo**:
   - Debug rapido con `coco8.yaml` (≈2 minuti).
   - Run intermedia 50 epoche per calibrare iperparametri.
   - Training finale `coco.yaml`, 100 epoche, batch 64 (≈8–12h su RTX 3060).
4. **Export**: `python scripts/export_models.py` genera `best.torchscript` e `best.onnx`.
5. **Benchmark**: `python scripts/benchmark_models.py` produce log comparativi su GPU/CPU (`benchmarks.log`).

### Output atteso
- Checkpoint `runs/detect/train/weights/best.pt` e grafici `results.png`.
- Modelli ottimizzati TorchScript (~12 MB) e ONNX (~25 MB).
- Tabelle prestazionali con precision, recall, mAP@0.5/0.5:0.95 e FPS.

---

## YOLOv8s – Post-Training Quantization (PTQ)
Directory: [`yoloV8s_PTQ/`](yoloV8s_PTQ/)  
Documentazione: [`yoloV8s_PTQ/README.md`](yoloV8s_PTQ/README.md)

### Scopo
Convertire il modello YOLOv8s addestrato su COCO2017 in un modello **TensorFlow Lite INT8** utilizzando la pipeline Ultralytics PTQ, mantenendo accuratezza compatibile con deployment edge.

### Organizzazione
```
yoloV8s_PTQ/
└── yolov8s_PTQ_INT8_Calib500/
    ├── calibration/                # Campioni e cache per representative dataset (500 immagini COCO val)
    ├── weights/                    # Checkpoint PyTorch di partenza (yolov8s.pt)
    └── tflite_exports/
        └── saved_model/            # SavedModel TF e variante quantizzata `yolov8s_int8.tflite`
```

### Pipeline di conversione
- Comando base: `yolo export model=yolov8s.pt format=tflite int8=True data=val2017_calib.yaml imgsz=640`.
- Representative dataset definito da `val2017_calib.yaml` (500 immagini COCO val).
- Tempo di conversione: ≈4 ore su CPU Apple M1 Pro.

### Risultati
- SavedModel 140 MB e modello TFLite INT8 da ≈11 MB.
- Riduzione dimensione ~4× con perdita minima di mAP (testato con `detect_tflite.py` su Raspberry Pi 4B: 4-5 FPS @320×240).
- Pronto per deployment su SBC, EdgeTPU o Android.

---

## EfficientDet INT8 – Benchmark
Directory: [`efficientdet_tests/`](efficientdet_tests/)  
Documentazione: [`efficientdet_tests/README.md`](efficientdet_tests/README.md)

### Contesto
Valutazione quantitativa dei modelli EfficientDet-D0 e D3 quantizzati INT8 su COCO 2017 (validation) tramite script dedicati di inferenza e calcolo metriche COCO.

### Cosa contiene
- `efficientdet_d0_int8/` e `efficientdet_d3_int8/` con:
  - `scripts/predizioniEBox*.py`: inferenza TFLite con pre/post-processing vettoriale e salvataggio JSON (debug + formato COCO).
  - `scripts/calcolo_mAP_valCOCO.py`: wrapper COCOeval con report estesi, grafici (main metrics, distribuzione categorie, heatmap dimensioni) e opzioni CLI.
  - `predictions/`: output delle predizioni.
  - `reports/`: JSON `detailed_evaluation_report.json` e grafici in `figures/`.
- `ModelliLiteTF/`: contiene i 2 modelli pre addestrati su COCO2017
### Riproduzione esperimenti
1. Lanciare gli script di inferenza passando percorsi al modello TFLite, annotazioni `instances_val2017.json` e immagini `val2017`.
2. Eseguire `calcolo_mAP_valCOCO.py` puntando al JSON COCO per generare metriche e grafici.

### Metriche sintetiche
- **EfficientDet-D0 INT8**: mAP@0.5:0.95 ≈ 0.25, mAP@0.5 ≈ 0.40.
- **EfficientDet-D3 INT8**: mAP@0.5:0.95 ≈ 0.36, mAP@0.5 ≈ 0.53, miglioramenti su tutte le scale di oggetto.

---
## Deployment su Raspberry Pi / SBC
Directory: [`raspberry_deployment/`](raspberry_deployment/)

### Contenuto
Questa sezione raccoglie tutti gli script e le configurazioni usate per eseguire
**modelli TFLite quantizzati (INT8)** su Raspberry Pi 4 e 5, testati in tempo reale
sia con **anteprima video (OpenCV)** sia in modalità **headless da terminale**.

### Componenti principali
- `object_detection_camera/`: esecuzione con anteprima video, FPS e bounding box.
- `object_detection_terminal/`: versione senza GUI per sessioni SSH o test remoti.
- `legacy/`: vecchie versioni di script YOLOv8/EfficientDet e utilità di supporto.
- `test_cameras.py`: utility per sondare rapidamente le webcam/CSI disponibili.
- Modelli supportati: **YOLOv8s INT8** e **EfficientDet Lite (D0–D3)**.

### Cosa è stato fatto
- Ottimizzazione delle pipeline di inferenza per aumentare gli FPS mantenendo
  l’accuratezza.
- Organizzazione delle cartelle e standardizzazione degli argomenti CLI.
- Aggiunta di script per entrambi i modelli (YOLOv8 e EfficientDet) con gestione
  di **quantizzazione input/output**, **threading** e **statistiche runtime**.
- Verifica prestazionale su Raspberry Pi 4 e 5 con note sui risultati (FPS medi,
  consumo CPU e stabilità camera).

### Come usarlo e Esepio
1. Creare un ambiente virtuale Python 3.7 e installare `tflite-runtime`, `opencv-python`, `numpy`.
2. Copiare i modelli `.tflite` nella cartella `~/models/` del Raspberry.
3. Eseguire lo script desiderato, ad esempio:
   ```bash
   python3 yolov8_int8_camera.py \
       --model ~/models/yolov8s-int8.tflite \
       --camera_id 0 --resolution 640x480 --num_threads 4

---

## Streaming UDP a bassa latenza
Directory: [`streaming_UDP/`](streaming_UDP/)  
Documentazione: [`streaming_UDP/README.md`](streaming_UDP/README.md)

### Scopo
Inviare i frame annotati (ad es. da EfficentDet-D3 TFLite) verso una postazione remota via UDP, mantenendo latenze minime.

### Script
- `UDPstriming/StrimingUDP/main.py`:
  - Interfaccia CLI per selezionare modello TFLite, camera, parametri UDP e thread.
  - Pipeline multi-thread: acquisizione video, inferenza TFLite (`tflite_support`), codifica JPEG, suddivisione in chunk con header custom e invio via UDP.
  - Buffer limitati (`Queue`) per ridurre la latenza, calcolo FPS e supporto EdgeTPU opzionale.
- `UDPstriming/StrimingUDP/receiver.py`:
  - Ricompone i chunk ricevuti usando identificativi di frame, decodifica JPEG e mostra i frame con OpenCV.
  - Timeout configurabile e gestione di pacchetti corrotti.

### Utilizzo tipico
1. Installare `opencv-python`, `numpy`, `tflite-support` su sender e `opencv-python`, `numpy` sul receiver.
2. Avviare il receiver: `python UDPstriming/StrimingUDP/receiver.py` (porta default 5200).
3. Avviare il sender specificando IP/porta della macchina di destinazione e il modello TFLite da utilizzare.

---

## Dataset e gestione dei dati
- La cartella `data/` è lasciata vuota per ospitare dataset locali (es. struttura COCO richiesta dal training YOLOv8s).
- I file di calibrazione PTQ e i dataset di inferenza EfficientDet vanno posizionati localmente e referenziati via CLI.
- Per motivi di licenza e dimensioni, i dataset COCO non sono inclusi; seguire le istruzioni nei README dedicati per il download.

---

## Riferimenti
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [COCO Dataset](https://cocodataset.org/)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [ONNX Runtime](https://onnxruntime.ai/)

---