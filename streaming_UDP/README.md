# UDP Streaming Utilities

Questa cartella contiene uno streaming UDP a bassa latenza per inviare i frame elaborati da un modello TensorFlow Lite verso un visualizzatore remoto. I due script principali si trovano in `UDPstriming/StrimingUDP/` e lavorano in coppia:

- `main.py` avvia l'acquisizione da camera, esegue l'object detection con TFLite e invia i frame annotati via UDP.
- `receiver.py` ricompone i frame ricevuti e li mostra in una finestra OpenCV.

## Dipendenze

Entrambi gli script richiedono Python 3 e le seguenti librerie:

- `opencv-python`
- `numpy`
- `tflite-support`

Installa le dipendenze con:

```bash
pip install opencv-python numpy tflite-support
```

## Sender (`main.py`)

Lo script espone una CLI per configurare il modello, la camera e i parametri UDP. Esegui, ad esempio:

```bash
python UDPstriming/StrimingUDP/main.py \
    --model efficientdet_lite0.tflite \
    --cameraId 0 \
    --frameWidth 640 \
    --frameHeight 480 \
    --udpHost 192.168.1.10 \
    --udpPort 5200 \
    --showLocal
```

### Parametri principali

| Flag | Descrizione |
|------|-------------|
| `--model` | Percorso del modello TFLite da utilizzare per l'object detection. |
| `--cameraId` | Indice della camera da usare (default 0). |
| `--frameWidth`, `--frameHeight` | Risoluzione desiderata del flusso in uscita. |
| `--numThreads` | Numero di thread CPU per il motore TFLite. |
| `--scoreThreshold` | Soglia di confidenza minima per visualizzare le detection. |
| `--udpHost`, `--udpPort` | Destinazione del flusso UDP. |
| `--showLocal` | Visualizza anche la finestra locale oltre allo streaming. |

Lo script ricerca automaticamente le camere disponibili, applica alcune ottimizzazioni (MJPEG, buffer ridotto) e trasmette i frame JPEG segmentati in chunk UDP.

## Receiver (`receiver.py`)

Avvia il ricevitore sulla macchina che deve mostrare lo stream:

```bash
python UDPstriming/StrimingUDP/receiver.py
```

Per impostazione predefinita ascolta su `0.0.0.0:5200`. Modifica `UDP_IP` e `UDP_PORT` nel file se vuoi ricevere su un indirizzo o porta differenti. Quando tutti i chunk di un frame sono stati ricevuti, il frame viene ricostruito e mostrato in una finestra OpenCV (`q` per uscire).

## Suggerimenti

1. Assicurati che sender e receiver possano comunicare sulla porta UDP scelta (apri il firewall se necessario).
2. Se noti frame drop, riduci la risoluzione o aumenta la qualità di compressione JPEG nello script sender.
3. Per test locali puoi usare `--udpHost 127.0.0.1` e avviare sender e receiver sulla stessa macchina.
