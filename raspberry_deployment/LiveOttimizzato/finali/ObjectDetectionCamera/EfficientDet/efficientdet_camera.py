import cv2
import numpy as np
import time
import argparse
import tflite_runtime.interpreter as tflite

# COCO class names per YOLOv8
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def non_max_suppression(boxes, scores, classes, iou_threshold=0.5):
    """Non-Maximum Suppression per eliminare le bounding box duplicate"""
    if len(boxes) == 0:
        return [], [], []
    
    # Converti in numpy arrays
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    classes = np.array(classes, dtype=np.int32)
    
    # Calcola le aree delle bounding box
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # Ordina per score decrescente
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
            
        # Calcola IoU con le altre box
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / (union + 1e-6)
        
        # Mantieni solo le box con IoU < threshold
        indices = np.where(iou <= iou_threshold)[0]
        order = order[indices + 1]
    
    return boxes[keep].tolist(), scores[keep].tolist(), classes[keep].tolist()

def preprocess_frame(frame, input_size):
    """Preprocessing ottimizzato del frame"""
    # Resize mantenendo aspect ratio se necessario
    input_img = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = input_img.astype(np.float32) / 255.0
    return np.expand_dims(input_img, axis=0)


# Argomenti da linea di comando
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to the .tflite model file')
parser.add_argument('--camera_id', type=int, default=0, help='Camera ID')
parser.add_argument('--imgsz', type=int, default=640, help='Inference image size (square)')
parser.add_argument('--conf_threshold', type=float, default=0.5, help='Confidence threshold')
parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for NMS')
parser.add_argument('--num_threads', type=int, default=4, help='Number of threads')
parser.add_argument('--skip_frames', type=int, default=2, help='Skip frames for better performance')
args = parser.parse_args()

# Inizializzazione modello

print("Caricamento modello...")
interpreter = tflite.Interpreter(model_path=args.model, num_threads=args.num_threads)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape: {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")


# Inizializzazione webcam con impostazioni ottimizzate

cap = cv2.VideoCapture(args.camera_id)

# Impostazioni ottimizzate per Raspberry Pi
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Risoluzione più bassa per migliori FPS
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer minimo per ridurre latency

# Variabili per il controllo dei frame
frame_count = 0
fps_counter = 0
fps_start_time = time.time()

print(" Inference running. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print(" Frame non disponibile.")
        break
    
    frame_count += 1
    
    # Skip frames per migliorare le performance
    if frame_count % (args.skip_frames + 1) != 0:
        cv2.imshow("YOLOv8 INT8 TFLite", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    # Preprocessing
    input_img = preprocess_frame(frame, args.imgsz)
    
    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_img)
    
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time
    
    # Post-processing
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    pred = output_data.transpose(1, 0)  # (8400, 84)
    
    boxes = []
    scores = []
    classes = []
    
    # Estrazione delle detection con soglia di confidenza
    for det in pred:
        # Per YOLOv8, le prime 4 sono coordinate, poi objectness + classi
        obj_conf = det[4]  # Object confidence
        
        if obj_conf > args.conf_threshold:
            # Trova la classe con score più alto
            class_scores = det[5:]  # Score delle classi
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]
            
            # Score finale = objectness * class_confidence
            final_score = obj_conf * class_conf
            
            if final_score > args.conf_threshold:
                # Coordinate (formato center_x, center_y, width, height)
                cx, cy, w, h = det[0], det[1], det[2], det[3]
                
                # Converti in pixel coordinates
                x1 = int((cx - w / 2) * frame.shape[1])
                y1 = int((cy - h / 2) * frame.shape[0])
                x2 = int((cx + w / 2) * frame.shape[1])
                y2 = int((cy + h / 2) * frame.shape[0])
                
                # Clamp alle dimensioni dell'immagine
                x1 = max(0, min(x1, frame.shape[1]))
                y1 = max(0, min(y1, frame.shape[0]))
                x2 = max(0, min(x2, frame.shape[1]))
                y2 = max(0, min(y2, frame.shape[0]))
                
                if x2 > x1 and y2 > y1:  # Verifica che la box sia valida
                    boxes.append([x1, y1, x2, y2])
                    scores.append(final_score)
                    classes.append(class_id)
    
    # Applica Non-Maximum Suppression
    if boxes:
        boxes, scores, classes = non_max_suppression(boxes, scores, classes, args.iou_threshold)
    
    # Drawing
    for box, score, cls_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        
        # Colore basato sulla classe
        color = (0, 255, 0) if cls_id < 10 else (255, 0, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Label con nome classe se disponibile
        if cls_id < len(COCO_CLASSES):
            label = f"{COCO_CLASSES[cls_id]}: {score:.2f}"
        else:
            label = f"Class {cls_id}: {score:.2f}"
        
        # Background per il testo
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Calcola FPS
    fps_counter += 1
    if time.time() - fps_start_time >= 1.0:
        fps = fps_counter / (time.time() - fps_start_time)
        fps_counter = 0
        fps_start_time = time.time()
    else:
        fps = fps_counter / (time.time() - fps_start_time + 1e-6)
    
    # Info display
    info_text = f"FPS: {fps:.1f} | Inference: {inference_time*1000:.1f}ms | Objects: {len(boxes)}"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.imshow("YOLOv8 INT8 TFLite", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(" Applicazione terminata.")