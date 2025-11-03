
# YOLOv8s Post-Training Quantization (PTQ)

Questa directory contiene i risultati finali della **quantizzazione post-training (PTQ)** del modello **YOLOv8s** realizzata con **Ultralytics YOLO**.
L’obiettivo del processo è stato convertire il modello addestrato su **COCO 2017** in una versione **TensorFlow Lite quantizzata in INT8**, ottimizzata per l’esecuzione su dispositivi a risorse limitate (es. **Raspberry Pi**, **Jetson Nano**, o altri SBC).

---

## Contenuto della directory

```
yolov8s_PTQ_INT8_Calib500/
├── calibration/              # Samples used for TensorFlow Lite representative dataset
├── weights/                  # Baseline PyTorch checkpoint taken as export input
└── tflite_exports/
    └── saved_model/          # SavedModel and TFLite variants produced from the PTQ run
```
### Descrizione sintetica

- **calibration/** → contiene i campioni rappresentativi e i metadati (`val2017_500_calib.yaml`) usati per la calibrazione del modello.  
- **weights/** → include il checkpoint PyTorch `yolov8s.pt` usato come base per la conversione.  
- **tflite_exports/** → contiene le versioni esportate del modello in diversi formati (`float32`, `float16`, `int8`), tra cui il file principale `yolov8s_int8.tflite`.

---

Tutti i modelli sono stati **testati localmente** e sono **pronti per l’esecuzione su dispositivi edge**, mantenendo un buon equilibrio tra accuratezza e prestazioni.
# Conversione del modello YOLOv8s in TensorFlow Lite INT8

## Obiettivo

Convertire il modello **YOLOv8s addestrato su COCO2017, addestrato in precedenza (`yolov8s.pt`)** in un modello **TensorFlow Lite quantizzato in INT8**, per l'esecuzione su dispositivi edge (es. Raspberry Pi o Jetson Nano), mantenendo un buon compromesso tra accuratezza e prestazioni.

---

## Processo di conversione

La conversione è stata effettuata tramite il comando ufficiale **Ultralytics YOLOv8 export**, che consente di esportare il modello PyTorch in diversi formati, tra cui TensorFlow Lite:

```bash
yolo export model=yolov8s.pt format=tflite int8=True data=val2017_calib.yaml imgsz=640
```

### Significato dei parametri principali

| Parametro | Descrizione                                                         |
|-----------|---------------------------------------------------------------------|
| `model=yolov8s.pt` | Modello YOLOv8s addestrato su COCO 2017                             |
| `format=tflite` | Formato di esportazione: TensorFlow Lite                            |
| `int8=True` | Abilita la quantizzazione a 8 bit (Post-Training Quantization - PTQ) |
| `data=val2017_calib.yaml` | File YAML contenente il percorso delle immagini di calibrazione     |
| `imgsz=640` | Dimensione di input delle immagini durante la conversione           |

---

## Dataset di calibrazione

- Utilizzate **500 immagini** provenienti da **COCO 2017 (val)**
- Percorso originale impiegato per la calibrazione:
  ```
  /Users/enrico/Desktop/YoloV8s/CocoVal500/
  ```
- File YAML di riferimento:
  ```yaml
  path: /Users/enrico/Desktop/YoloV8s/CocoVal500
  val: images
  names:
    0: person
    1: bicycle
    2: car
    ...
    79: toothbrush
  ```

Durante la conversione, i log mostravano se non si utilizzano almeno 500 immagini per la calibrazione:

```
TensorFlow SavedModel: collecting INT8 calibration images from 'data=val2017_calib.yaml'
Fast image access 
WARNING ️ Labels are missing or empty in calibration cache
WARNING  >300 images recommended for INT8 calibration, found 500 images.
```

---

## Risultato dell'esportazione

Durata: circa **4 ore** su CPU Apple M1 Pro.

Output generato e conservato nella cartella `yolov8s_PTQ_INT8_Calib500/`:
```
yolov8s_PTQ_INT8_Calib500/
├── calibration/
│   └── calibration_image_sample_data_20x128x128x3_float32.npy
└── tflite_exports/
    └── saved_model/
        ├── assets/
        ├── variables/
        ├── saved_model.pb
        └── yolov8s_int8.tflite     ← modello finale quantizzato INT8 (≈11 MB)
```

Messaggio finale:
```
TensorFlow SavedModel: export success (140.0 MB)
TensorFlow Lite: export success (11.0 MB)
Export complete.
```

---

## Osservazioni

- La quantizzazione **INT8 PTQ** riduce le dimensioni (42.9 MB → 11 MB) con minima perdita di accuratezza
- Compatibile con **TFLite Runtime** (Raspberry Pi, Jetson Nano, Android Edge TPU)
- Input e output in FLOAT32, pesi INT8
- funziona in real-time su SBC
- mantiene compatibilità COCO 2017 (80 classi)
- è stato quantizzato INT8 con 500 immagini di calibrazione

