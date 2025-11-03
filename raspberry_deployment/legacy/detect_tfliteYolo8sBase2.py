import cv2
import numpy as np
import time
import argparse
import tflite_runtime.interpreter as tflite
from collections import deque
from threading import Thread, Lock
import queue

class FrameBuffer:
    """Buffer thread-safe per frame processing asincrono"""
    def __init__(self, maxsize=2):
        self.buffer = queue.Queue(maxsize=maxsize)
        self.lock = Lock()
    
    def put(self, frame):
        try:
            self.buffer.put_nowait(frame)
        except queue.Full:
            # Rimuovi frame vecchio e inserisci nuovo
            try:
                self.buffer.get_nowait()
                self.buffer.put_nowait(frame)
            except queue.Empty:
                pass
    
    def get(self):
        try:
            return self.buffer.get_nowait()
        except queue.Empty:
            return None

def vectorized_nms(boxes, scores, iou_threshold=0.45):
    """NMS completamente vettorizzato - più veloce"""
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    
    # Calcola aree una sola volta
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # Ordina per score
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
        
        # Calcola IoU vettorizzato per tutti i box rimanenti
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        
        # Mantieni solo box con IoU basso
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep

class OptimizedYOLODetector:
    def __init__(self, model_path, num_threads=4, conf_threshold=1e-6, 
                 iou_threshold=0.45, min_area=100):
        # Parametri
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_area = min_area
        
        # Classi COCO ottimizzate
        self.classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus',
            7: 'truck', 15: 'cat', 16: 'dog', 39: 'bottle', 56: 'chair',
            57: 'couch', 62: 'tv', 63: 'laptop', 67: 'cell phone'
        }
        
        # Colori per categoria
        self.colors = {
            0: (255, 100, 100),  # person - rosso chiaro
            2: (100, 100, 255), 3: (100, 100, 255), 5: (100, 100, 255), 7: (100, 100, 255),  # veicoli - blu
        }
        self.default_color = (100, 255, 100)  # verde chiaro
        
        # Inizializza modello
        print("🔄 Caricamento modello ottimizzato...")
        self.interpreter = tflite.Interpreter(
            model_path=model_path, 
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Pre-alloca array per evitare allocazioni durante inference
        self.input_shape = self.input_details[0]['shape']
        self.input_tensor = np.zeros(self.input_shape, dtype=np.float32)
        
        print(f"✅ Modello caricato! Input: {self.input_shape}")
        
        # Buffer per stabilizzazione detection
        self.detection_buffer = deque(maxlen=3)
        self.stable_threshold = 40  # Distanza massima tra centri per considerare stabile
        
    def preprocess_optimized(self, frame, target_size):
        """Preprocessing ottimizzato con allocazioni minime"""
        # Resize con interpolazione lineare (più veloce)
        resized = cv2.resize(frame, (target_size, target_size), 
                           interpolation=cv2.INTER_LINEAR)
        
        # Conversione colore e normalizzazione in un passaggio
        rgb_normalized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb_normalized *= (1.0 / 255.0)  # Più veloce di divisione
        
        # Copia direttamente nel tensor pre-allocato
        self.input_tensor[0] = rgb_normalized
        return self.input_tensor
    
    def postprocess_optimized(self, output_data, frame_shape):
        """Post-processing ottimizzato con operazioni vettoriali"""
        predictions = output_data.transpose(1, 0)  # (8400, 84)
        
        # Estrai coordinate e confidence in batch
        coords = predictions[:, :4]  # cx, cy, w, h
        class_confs = predictions[:, 4:]
        
        # Trova classe migliore per ogni prediction (vettorizzato)
        max_conf_indices = np.argmax(class_confs, axis=1)
        max_confs = class_confs[np.arange(len(class_confs)), max_conf_indices]
        
        # Filtra per confidence minima
        valid_mask = max_confs > self.conf_threshold
        if not np.any(valid_mask):
            return [], [], []
        
        valid_coords = coords[valid_mask]
        valid_confs = max_confs[valid_mask]
        valid_classes = max_conf_indices[valid_mask]
        
        # Converti coordinate (vettorizzato)
        h, w = frame_shape[:2]
        cx, cy, bw, bh = valid_coords.T
        
        x1 = ((cx - bw/2) * w).astype(np.int32)
        y1 = ((cy - bh/2) * h).astype(np.int32)
        x2 = ((cx + bw/2) * w).astype(np.int32)
        y2 = ((cy + bh/2) * h).astype(np.int32)
        
        # Clamp coordinate
        x1 = np.clip(x1, 0, w-1)
        y1 = np.clip(y1, 0, h-1)
        x2 = np.clip(x2, 0, w-1)
        y2 = np.clip(y2, 0, h-1)
        
        # Filtra per area minima
        areas = (x2 - x1) * (y2 - y1)
        area_mask = (areas > self.min_area) & (x2 > x1) & (y2 > y1)
        
        if not np.any(area_mask):
            return [], [], []
        
        final_boxes = np.column_stack([x1, y1, x2, y2])[area_mask]
        final_scores = valid_confs[area_mask]
        final_classes = valid_classes[area_mask]
        
        return final_boxes.tolist(), final_scores.tolist(), final_classes.tolist()
    
    def stabilize_detections(self, current_detections):
        """Stabilizza detection tra frame consecutivi"""
        if not current_detections:
            self.detection_buffer.append([])
            return []
        
        self.detection_buffer.append(current_detections)
        
        if len(self.detection_buffer) < 2:
            return current_detections
        
        # Trova detection stabili (presenti in frame precedente)
        stable_detections = []
        current_boxes, current_scores, current_classes = zip(*current_detections)
        prev_detections = self.detection_buffer[-2]
        
        if not prev_detections:
            return current_detections
        
        prev_boxes, prev_scores, prev_classes = zip(*prev_detections)
        
        # Calcola centri (vettorizzato)
        current_boxes = np.array(current_boxes)
        prev_boxes = np.array(prev_boxes)
        
        curr_centers = (current_boxes[:, [0, 2]].mean(axis=1), 
                       current_boxes[:, [1, 3]].mean(axis=1))
        prev_centers = (prev_boxes[:, [0, 2]].mean(axis=1), 
                       prev_boxes[:, [1, 3]].mean(axis=1))
        
        # Trova corrispondenze
        for i, (curr_cls, curr_score) in enumerate(zip(current_classes, current_scores)):
            curr_center = (curr_centers[0][i], curr_centers[1][i])
            
            for j, prev_cls in enumerate(prev_classes):
                if curr_cls == prev_cls:
                    prev_center = (prev_centers[0][j], prev_centers[1][j])
                    dist = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                 (curr_center[1] - prev_center[1])**2)
                    
                    if dist < self.stable_threshold:
                        stable_detections.append(current_detections[i])
                        break
        
        return stable_detections if stable_detections else current_detections[:5]  # Massimo 5 se non stabili
    
    def detect(self, frame):
        """Detection principale ottimizzata"""
        # Preprocessing
        input_tensor = self.preprocess_optimized(frame, self.input_shape[1])
        
        # Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        
        start_inference = time.time()
        self.interpreter.invoke()
        inference_time = time.time() - start_inference
        
        # Post-processing
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        boxes, scores, classes = self.postprocess_optimized(output_data, frame.shape)
        
        # NMS
        if boxes:
            keep_indices = vectorized_nms(boxes, scores, self.iou_threshold)
            filtered_detections = [(boxes[i], scores[i], classes[i]) for i in keep_indices[:8]]  # Max 8
            
            # Stabilizzazione
            stable_detections = self.stabilize_detections(filtered_detections)
            return stable_detections, inference_time
        
        return [], inference_time
    
    def draw_detections(self, frame, detections):
        """Drawing ottimizzato"""
        for box, score, cls_id in detections:
            x1, y1, x2, y2 = map(int, box)
            
            # Colore basato su classe
            color = self.colors.get(cls_id, self.default_color)
            
            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label ottimizzato
            label = self.classes.get(cls_id, f"ID{cls_id}")
            text = f"{label}:{score:.2f}"
            
            # Background per testo
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            cv2.rectangle(frame, (x1, y1 - th - baseline - 2), (x1 + tw, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - baseline - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

def main():
    # Argomenti
    parser = argparse.ArgumentParser(description="YOLOv8 TensorFlow Lite Ottimizzato")
    parser.add_argument('--model', required=True, help='Path al modello .tflite')
    parser.add_argument('--camera_id', type=int, default=0, help='ID Camera')
    parser.add_argument('--imgsz', type=int, default=640, help='Dimensione input')
    parser.add_argument('--num_threads', type=int, default=4, help='Thread CPU')
    parser.add_argument('--skip_frames', type=int, default=2, help='Frame da saltare')
    parser.add_argument('--conf_threshold', type=float, default=1e-6, help='Soglia confidence')
    parser.add_argument('--iou_threshold', type=float, default=0.45, help='Soglia IoU per NMS')
    parser.add_argument('--resolution', type=str, default='320x240', help='Risoluzione camera (WxH)')
    args = parser.parse_args()
    
    # Parsing risoluzione
    width, height = map(int, args.resolution.split('x'))
    
    # Inizializza detector
    detector = OptimizedYOLODetector(
        model_path=args.model,
        num_threads=args.num_threads,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )
    
    # Inizializza camera con buffer ottimizzato
    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer minimo
    
    # Metriche performance
    frame_count = 0
    fps_buffer = deque(maxlen=30)
    inference_buffer = deque(maxlen=30)
    last_detections = []
    
    print(f"Avvio detection ottimizzata ({width}x{height})")
    print("Premi 'q' per uscire, 's' per statistiche")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Errore lettura frame")
                break
            
            frame_count += 1
            start_total = time.time()
            
            # Skip frame logic migliorata
            if frame_count % (args.skip_frames + 1) == 0:
                # Esegui detection
                detections, inference_time = detector.detect(frame)
                last_detections = detections
                inference_buffer.append(inference_time)
            else:
                # Usa ultima detection
                detections = last_detections
                inference_time = inference_buffer[-1] if inference_buffer else 0
            
            # Drawing
            detector.draw_detections(frame, detections)
            
            # Calcola FPS
            total_time = time.time() - start_total
            current_fps = 1.0 / total_time if total_time > 0 else 0
            fps_buffer.append(current_fps)
            
            # Info overlay ottimizzato
            avg_fps = np.mean(fps_buffer)
            avg_inference = np.mean(inference_buffer) if inference_buffer else 0
            
            info_y = 25
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Inference: {avg_inference*1000:.0f}ms", (10, info_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Objects: {len(detections)}", (10, info_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow("YOLOv8 Ottimizzato", frame)
            
            # Input handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print(f"\n📊 Statistiche:")
                print(f"   FPS medio: {avg_fps:.2f}")
                print(f"   Inference media: {avg_inference*1000:.1f}ms")
                print(f"   Frame processati: {frame_count}")
    
    except KeyboardInterrupt:
        print("\n⏹Interruzione utente")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if fps_buffer:
            final_fps = np.mean(fps_buffer)
            print(f"   Performance finale:")
            print(f"   FPS medio: {final_fps:.2f}")
            print(f"   Frame totali: {frame_count}")
        
        print("Rilascio risorse completato")

if __name__ == "__main__":
    main()