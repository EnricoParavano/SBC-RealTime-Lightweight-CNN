import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark


# Config
MODEL_ARCH = 'yolov8s.pt'
DATASET = 'coco.yaml'
EPOCHS = 100
BATCH = 64
IMG_SIZE = 640
DEVICE = 0  # GPU = 0, CPU = 'cpu'
RUN_NAME = 'benchmark_yolov8s_full_100'
PATIENCE = 10  # early stopping: numero massimo di epoche senza miglioramento

def early_stopping_check(csv_path, patience):
    df = pd.read_csv(csv_path)
    map_key = 'metrics/mAP_0.5:0.95'
    if map_key not in df.columns or len(df) <= patience:
        return False  # non abbastanza dati

    best = -1
    count = 0
    for value in df[map_key]:
        if value > best:
            best = value
            count = 0
        else:
            count += 1
        if count >= patience:
            return True
    return False

def get_latest_run(path='runs/detect/'):
    runs = sorted(
        [d for d in glob.glob(os.path.join(path, '*')) if os.path.isdir(d)],
        key=os.path.getmtime
    )
    return runs[-1] if runs else None

def main():
    print(f"[INFO] Inizio addestramento modello {MODEL_ARCH}...")
    model = YOLO(MODEL_ARCH)
    model.train(
        data=DATASET,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMG_SIZE,
        device=DEVICE,
        # non serve name=, così YOLO genera il nome dinamico
    )

    run_dir = get_latest_run()
    if not run_dir:
        print("[ERRORE] Nessuna directory di run trovata.")
        return

    model_path = os.path.join(run_dir, 'weights', 'best.pt')
    results_csv_path = os.path.join(run_dir, 'results.csv')

    # Validazione
    print("[INFO] Fine training. Inizio validazione...")
    model.val(
        model=model_path,
        data=DATASET,
        imgsz=IMG_SIZE,
        device=DEVICE,
        save_dir=run_dir,
        save_json=True
    )

    # Grafico
    if os.path.exists(results_csv_path):
        print("[INFO] Visualizzazione dei grafici...")
        df = pd.read_csv(results_csv_path)
        metrics_to_plot = [
            "train/box_loss", "train/cls_loss", "train/dfl_loss",
            "metrics/precision", "metrics/recall", "metrics/mAP_0.5", "metrics/mAP_0.5:0.95"
        ]
        plt.figure(figsize=(12, 6))
        for metric in metrics_to_plot:
            if metric in df.columns:
                plt.plot(df[metric], label=metric)
        plt.title("Metriche di Addestramento e Validazione YOLOv8")
        plt.xlabel("Epoca")
        plt.ylabel("Valore")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)  # non blocca lo script
    else:
        print(f"[ERRORE] results.csv non trovato in {results_csv_path}")

    # BENCHMARK
    print("[INFO] Avvio benchmark finale...")
    benchmark(
        model=model_path,
        data=DATASET,
        imgsz=IMG_SIZE,
        half=False,
        device=DEVICE
    )


if __name__ == "__main__":
    main()
