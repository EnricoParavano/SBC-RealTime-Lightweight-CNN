# Deploy di object detection su Raspberry Pi

Questa cartella raccoglie tutti gli script usati per eseguire modelli TensorFlow Lite
su Raspberry Pi. I file sono stati raggruppati per modalità di utilizzo, così da
trovare subito ciò che serve sia per l'anteprima video sia per l'inferenza da
terminal.

## Struttura della cartella

- **`object_detection_camera/`** – script che aprono una finestra OpenCV con il
  flusso video annotato (bounding box, FPS, ecc.).
- **`object_detection_terminal/`** – versione "headless" pensata per sessioni SSH:
  usa la camera ma stampa solo i risultati nel terminale.
- **`legacy/`** – contiene gli script storici, non sono indispensabili per i workflow attuali, ma possono
  tornare utili come riferimento (per esempio le vecchie utility o il test della
  camera).


## 1. Preparazione del Raspberry Pi
### 1.1 Crea e attiva un ambiente virtuale (consigliato)
Gli appunti originali suggeriscono questo flusso, molto comodo sul Raspberry:
```bash
python3.7 -m venv effdet_env
source effdet_env/bin/activate
pip install --upgrade pip
pip install opencv-python tflite-support
```
Il pacchetto `opencv-python` può essere sostituito da `sudo apt install -y
python3-opencv` se preferisci usare le librerie precompilate di Raspberry Pi OS.

### 1.2 Installa le dipendenze TFLite
Se vuoi un set completo (incluso `tflite-runtime`), aggiungi:
```bash
pip install numpy==1.24.4 tflite-runtime==2.12.0
```
Regola le versioni in base al modello che vuoi eseguire. Tutti gli script
supportano il parametro `--num_threads` per sfruttare i core disponibili.

### 1.3 Copia modelli e script sul dispositivo
Puoi trasferire i file dal tuo PC con `scp`. Esempio preso dagli appunti:
```bash
scp "C:\Users\Enrico Paravano\Desktop\LiveOttimizzatoRaspberry\detect_optimized.py" \
    pi@192.168.0.72:/home/pi/Desktop/EfficentDetLiteLive/LiveOttimizzato/
```
Adatta il percorso remoto a questa nuova struttura (`~/object_detection_camera/`,
`~/object_detection_terminal/`, ecc.).

### 1.4 Verifica che la camera funzioni
Nel caso serva un rapido check degli ID disponibili:
```bash
python3 legacy/test_cameras.py
```
Il programma prova gli indici da 0 a 9 e stampa quelli validi.

---

## 2. Modalità con anteprima video (OpenCV)
Richiedono una sessione grafica attiva (monitor collegato, VNC oppure X11
forwarding). Chiudi con il tasto `q`.

### `object_detection_camera/yoloV8sINT8/yolov8_int8_camera.py`
Pipeline ottimizzata per modelli YOLOv8 INT8 con gestione del buffering per FPS
più alti.
```bash
python3 yolov8_int8_camera.py \
    --model /home/pi/models/yolov8s-int8.tflite \
    --camera_id 0 \
    --resolution 640x480 \
    --num_threads 4 \
    --skip_frames 2
```
Opzioni utili: `--conf_threshold`, `--iou_threshold`, `--show_detailed_stats`.

### `object_detection_camera/yoloV8sINT8Full/yolov8_int8_quant_camera.py`
Versione completa con gestione dei parametri di quantizzazione input/output.
```bash
python3 yolov8_int8_quant_camera.py \
    --model /home/pi/models/yolov8s-int8.tflite \
    --camera_id 0 \
    --resolution 640x480 \
    --num_threads 4
```
Puoi aggiungere `--conf_threshold`, `--iou_threshold` e `--show_fps`.

### `object_detection_camera/efficentDet/efficientdet_camera.py`
Baseline EfficientDet Lite, semplice da adattare a modelli diversi.
```bash
python3 efficientdet_camera.py \
    --model /home/pi/models/efficientdet_lite0.tflite \
    --camera_id 0 \
    --imgsz 320 \
    --skip_frames 2
```
Mostra bounding box, etichette e statistiche sul terminale.

---

## 3. Modalità solo terminale
Perfetta per sessioni SSH: nessuna finestra, ma stampa le detection e gli FPS.

### `object_detection_terminal/efficientdet_terminal.py`
```bash
python3 efficientdet_terminal.py \
    --model /home/pi/models/efficientdet_lite0.tflite \
    --cameraId 0 \
    --frameWidth 640 --frameHeight 480 \
    --numThreads 4 \
    --scoreThreshold 0.3
```
Argomenti aggiuntivi: `--maxFrames` per fermarsi automaticamente,
`--inputWidth`/`--inputHeight` per forzare il resize dell'immagine.

---

## 4. Cartella `legacy/`
Contiene tutti i file che erano nella radice di `LiveOttimizzato/`:

- `detect_tfliteYolo8sBase2.py` e `detect_tfliteYolo8sINT8.py`: vecchie prove
  con YOLOv8 (richiedono `legacy/utils.py`).
- `utils.py` e `utils_optimized.py`: helper per NMS e post-processing.
- `test_cameras.py`: utility per sondare rapidamente gli ID delle camere.

Puoi eliminarla dal deploy finale se ti serve solo l'attuale pipeline, ma
conservarla aiuta quando vuoi riattivare script sperimentali o copiare funzioni
utility.

---

## 5. Note sulle prestazioni (appunti )
- Modello **EfficientDet Lite D0 INT8** (`EfficientDet-D0.tflite`) con risoluzione 320×240 e
  `--numThreads 4`: ~9,5 FPS sul Raspberry dopo le ottimizzazioni.
- Modello **EfficientDet Lite D3 INT8** (`EfficientDet-D3.tflite`) a 512×512: FPS
  simili a prima dell'ottimizzazione (~4 FPS).
