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
