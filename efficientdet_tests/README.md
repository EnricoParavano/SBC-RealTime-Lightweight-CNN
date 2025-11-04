# EfficientDet INT8 - Test e Report

Questa cartella raccoglie gli esperimenti svolti con i modelli EfficientDet-D0 e EfficientDet-D3
quantizzati in INT8 e pre-addestrati su COCO 2017. Entrambi i modelli sono stati utilizzati per
generare predizioni sul validation set COCO e per calcolare le metriche standard di valutazione COCO.

## Struttura della cartella

```
efficientdet_tests/
├── README.md
├── efficientdet_d0_int8/
│   ├── predictions/                 # Predizioni raw e file COCO-ready
│   ├── reports/
│   │   ├── detailed_evaluation_report.json
│   │   └── figures/                 # Grafici riassuntivi (mAP, distribuzioni, heatmap)
│   └── scripts/
│       ├── predizioniEBoxD0.py      # Pipeline di inferenza e generazione predizioni
│       └── calcolo_mAP_valCOCO.py   # Script di valutazione e generazione grafici
├── efficientdet_d3_int8/
│    ├── predictions/
│    ├── reports/
│    │   ├── detailed_evaluation_report.json
│    │   └── figures/
│    └── scripts/
│       ├── predizioniEBoxD3.py
│       └── calcolo_mAP_valCOCO.py
└── MoswlliLiteTF
     ├── EfficientDet-D0.tflite
     └── EfficientDet-D3.tflite
```

> Nota: gli script di inferenza creano automaticamente la cartella `reports/qualitative_samples`
> (non tracciata in git) per salvare le prime immagini annotate.
### Modelli pre-addestrati

I modelli **EfficientDet-Lite0** e **EfficientDet-Lite3** utilizzati in questi test provengono da **Kaggle Models** e sono già **pre-addestrati sul dataset COCO 2017** e **quantizzati in formato INT8** per TensorFlow Lite.  
Possono essere scaricati dal seguente link ufficiale:

Link: [TensorFlow EfficientDet TFLite Models – Kaggle](https://www.kaggle.com/models/tensorflow/efficientdet/tfLite/lite0-int8)

Questi modelli includono pesi compatibili con l’inferenza su dispositivi embedded (come Raspberry Pi o Jetson Orin Nano) e possono essere utilizzati direttamente come input negli script di test di questa cartella.
## Come riprodurre i risultati

1. **Configura i percorsi locali** passando agli script i parametri `--model`, `--annotations`
   e `--images`. I valori di default corrispondono ai percorsi utilizzati negli esperimenti
   originali su Windows, ma possono essere sovrascritti da linea di comando senza modificare i file.
2. **Esegui l'inferenza** per generare le predizioni e i JSON di debug:

   ```bash
   cd efficientdet_tests/efficientdet_d0_int8/scripts
   python predizioniEBoxD0.py \
       --model /percorso/al/modello_d0.tflite \
       --annotations /percorso/alle/annotazioni/instances_val2017.json \
       --images /percorso/alle/immagini/val2017

   cd ../../efficientdet_d3_int8/scripts
   python predizioniEBoxD3.py \
       --model /percorso/al/modello_d3.tflite \
       --annotations /percorso/alle/annotazioni/instances_val2017.json \
       --images /percorso/alle/immagini/val2017
   ```

   Gli script produrranno:
   - `predictions/predictions_debug.json` con statistiche e bounding box dettagliate;
   - `predictions/predictions_coco_format.json` pronto per la valutazione COCO;
   - eventuali immagini annotate in `reports/qualitative_samples/`.

3. **Calcola le metriche COCO** e genera i grafici di riepilogo:

   ```bash
   python calcolo_mAP_valCOCO.py --predictions ../predictions/predictions_coco_format.json
   ```

   I risultati sono salvati in `reports/detailed_evaluation_report.json` e i grafici in
   `reports/figures/` (`main_metrics.png`, `top_categories_map.png`, `prediction_distribution.png`,
   `size_performance_heatmap.png`).

### Parametri principali

Gli script espongono opzioni CLI per adattarsi a diversi layout di progetto o dataset.

**Inferenza (`predizioniEBox*.py`)**

- `--model`, `--annotations`, `--images`: percorsi al modello TFLite, alle annotazioni COCO
  e alla cartella delle immagini.
- `--num-images`: numero massimo di immagini da elaborare (<=0 per tutte).
- `--save-images`: quante immagini annotate salvare in `reports/qualitative_samples/`.
- `--score-threshold`: soglia minima di confidenza.
- `--input-size`: dimensione del lato input (320 per D0, 512 per D3).
- `--category-offset`: offset per riallineare gli ID di classe alle categorie COCO.
- `--output-root`: cartella in cui salvare automaticamente `predictions/` e `reports/`.
- `--output-json` / `--output-coco-json`: percorsi personalizzati per i file JSON prodotti.
- Log aggiuntivi indicano i percorsi effettivi utilizzati e un riepilogo delle statistiche finali.

**Valutazione (`calcolo_mAP_valCOCO.py`)**

- `--predictions` / `--ground-truth`: file JSON delle predizioni e annotazioni COCO.
- `--output-root`: cartella base per `reports/` e `figures/` (utile per eseguire gli script fuori
  dal repository).
- `--report-path` / `--figures-dir`: percorsi completamente personalizzati per il report e i grafici.
- `--top-k-categories`: numero di categorie mostrate nei grafici comparativi.
- `--no-visualizations`: disabilita la generazione dei grafici mantenendo il solo report JSON.

## Sintesi delle performance

- **EfficientDet-D0 INT8**: mAP@0.5:0.95 ≈ 0.248, mAP@0.5 ≈ 0.400, con buone performance sugli oggetti
di dimensione medio-grande.
- **EfficientDet-D3 INT8**: mAP@0.5:0.95 ≈ 0.360, mAP@0.5 ≈ 0.529, con miglioramenti diffusi su tutte
le dimensioni grazie al backbone più profondo.

Entrambi i report contengono analisi di distribuzione delle categorie, statistiche sui punteggi e
le prime 20 classi ordinate per mAP.
