# Datasets usati: COCO8 e COCO2017 

## COCO8 (mini per test)
**Cos’è**  
Micro-subset di 8 immagini (4 train, 4 val) derivato da COCO2017. Serve per test rapidi della pipeline (sanity check).

**Struttura (YOLO format)**
```
coco8/
├─ images/
│  ├─ train/    # 4 immagini
│  └─ val/      # 4 immagini
└─ labels/
   ├─ train/    # .txt YOLO: class x_c y_c w h (normalizzati)
   └─ val/
```

**Download**
- ZIP pronto (Ultralytics assets): https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip
- Documentazione: https://docs.ultralytics.com/datasets/detect/coco8/

## COCO2017 (standard completo, 80 classi)
**Cos’è**  
Dataset di riferimento per object detection. Split principali: `train2017` (~118k), `val2017` (~5k), `test2017` (opz., ~41k, senza label pubbliche). Annotazioni in formato COCO JSON.

**Struttura (formato originale COCO)**
```
coco2017/
├─ train2017/                     # immagini train
├─ val2017/                       # immagini val
├─ test2017/      (opzionale)     # immagini test (niente GT pubblico)
└─ annotations/
   ├─ instances_train2017.json
   ├─ instances_val2017.json
   ├─ captions_* (opz.)
   ├─ person_keypoints_* (opz.)
   └─ image_info_test2017.json
```

**Download (sito ufficiale COCO)**
- Immagini:
  - Train: http://images.cocodataset.org/zips/train2017.zip
  - Val:   http://images.cocodataset.org/zips/val2017.zip
  - (Opz.) Test: http://images.cocodataset.org/zips/test2017.zip
- Annotazioni (JSON):
  - http://images.cocodataset.org/annotations/annotations_trainval2017.zip
- Link Kaggle:
  - https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset
## Note veloci
- Con Ultralytics puoi usare anche `coco.yaml`, che scarica **immagini + label YOLO pronte** senza conversioni.
