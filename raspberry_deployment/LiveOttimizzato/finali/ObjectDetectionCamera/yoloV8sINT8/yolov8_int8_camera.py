import cv2
import numpy as np
import time
import argparse
import tflite_runtime.interpreter as tflite
from collections import deque
from threading import Thread, Lock
import queue
import time

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
    def __init__(self, model_path, num_threads=4, conf_threshold=0.25, 
                 iou_threshold=0.45, min_area=100):
        # Parametri
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_area = min_area
        
        # Classi COCO ottimizzate - solo quelle più comuni
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
        print(" Caricamento modello ottimizzato...")
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
        
        print(f" Modello caricato! Input: {self.input_shape}")
        print(f" Input dtype: {self.input_details[0]['dtype']}")
        print(f" Output dtype: {self.output_details[0]['dtype']}")
        
        # Buffer per stabilizzazione detection - ridotto per performance
        self.detection_buffer = deque(maxlen=2)
        self.stable_threshold = 50  # Distanza massima tra centri per considerare stabile
        
    def preprocess_optimized(self, frame, target_size):
        """Preprocessing ottimizzato con allocazioni minime"""
        # Resize con interpolazione lineare (più veloce)
        resized = cv2.resize(frame, (target_size, target_size), 
                           interpolation=cv2.INTER_LINEAR)
        
        # Conversione colore e normalizzazione in un passaggio
        rgb_normalized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb_normalized /= 255.0
        
        # Copia direttamente nel tensor pre-allocato
        self.input_tensor[0] = rgb_normalized
        return self.input_tensor
    
    def postprocess_optimized(self, output_data, frame_shape):
        """Post-processing ottimizzato con operazioni vettoriali"""
        # Gestisce sia shape (1, 84, 8400) che (1, 8400, 84)
        if output_data.shape[-1] == 84:
            predictions = output_data.squeeze(0)  # (8400, 84)
        else:
            predictions = output_data.squeeze(0).transpose(1, 0)  # (8400, 84)
        
        # Estrai coordinate e confidence in batch
        coords = predictions[:, :4]  # cx, cy, w, h
        class_confs = predictions[:, 4:]
        
        # Trova classe migliore per ogni prediction
        max_conf_indices = np.argmax(class_confs, axis=1)
        max_confs = class_confs[np.arange(len(class_confs)), max_conf_indices]
        
        # Filtra per confidence minima e solo classi conosciute
        valid_mask = (max_confs > self.conf_threshold) & np.isin(max_conf_indices, list(self.classes.keys()))
        
        if not np.any(valid_mask):
            return [], [], []
        
        valid_coords = coords[valid_mask]
        valid_confs = max_confs[valid_mask]
        valid_classes = max_conf_indices[valid_mask]
        
        # Converti coordinate
        h, w = frame_shape[:2]
        cx, cy, bw, bh = valid_coords.T
        
        x1 = np.clip(((cx - bw/2) * w).astype(np.int32), 0, w-1)
        y1 = np.clip(((cy - bh/2) * h).astype(np.int32), 0, h-1)
        x2 = np.clip(((cx + bw/2) * w).astype(np.int32), 0, w-1)
        y2 = np.clip(((cy + bh/2) * h).astype(np.int32), 0, h-1)
        
        # Filtra per area minima e validità
        areas = (x2 - x1) * (y2 - y1)
        area_mask = (areas > self.min_area) & (x2 > x1) & (y2 > y1)
        
        if not np.any(area_mask):
            return [], [], []
        
        final_boxes = np.column_stack([x1, y1, x2, y2])[area_mask]
        final_scores = valid_confs[area_mask]
        final_classes = valid_classes[area_mask]
        
        return final_boxes.tolist(), final_scores.tolist(), final_classes.tolist()
    
    def stabilize_detections(self, current_detections):
        """Stabilizza detection tra frame consecutivi - versione semplificata"""
        if not current_detections:
            self.detection_buffer.append([])
            return []
        
        self.detection_buffer.append(current_detections)
        
        # Per ora restituisce semplicemente le detection correnti limitate
        # La stabilizzazione completa può essere costosa
        return current_detections[:6]  # Massimo 6 oggetti per performance
    
    def detect(self, frame):
        """Detection principale ottimizzata"""
        # Preprocessing
        start_preprocess = time.time()
        input_tensor = self.preprocess_optimized(frame, self.input_shape[1])
        preprocess_time = time.time() - start_preprocess
        
        # Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        
        start_inference = time.time()
        self.interpreter.invoke()
        inference_time = time.time() - start_inference
        
        # Post-processing
        start_postprocess = time.time()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        boxes, scores, classes = self.postprocess_optimized(output_data, frame.shape)
        postprocess_time = time.time() - start_postprocess
        
        # NMS
        start_nms = time.time()
        if boxes:
            keep_indices = vectorized_nms(boxes, scores, self.iou_threshold)
            filtered_detections = [(boxes[i], scores[i], classes[i]) for i in keep_indices[:6]]  # Max 6
            
            # Stabilizzazione semplificata
            stable_detections = self.stabilize_detections(filtered_detections)
            nms_time = time.time() - start_nms
            
            total_processing_time = preprocess_time + inference_time + postprocess_time + nms_time
            
            return stable_detections, {
                'total': total_processing_time,
                'inference': inference_time,
                'preprocess': preprocess_time,
                'postprocess': postprocess_time,
                'nms': nms_time
            }
        
        nms_time = time.time() - start_nms
        total_processing_time = preprocess_time + inference_time + postprocess_time + nms_time
        
        return [], {
            'total': total_processing_time,
            'inference': inference_time,
            'preprocess': preprocess_time,
            'postprocess': postprocess_time,
            'nms': nms_time
        }
    
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

class FPSCounter:
    """Contatore FPS accurato"""
    def __init__(self, buffer_size=30):
        self.buffer = deque(maxlen=buffer_size)
        self.last_time = time.time()
    
    def update(self):
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.buffer.append(fps)
        self.last_time = current_time
        return fps
    
    def get_average(self):
        return np.mean(self.buffer) if self.buffer else 0

def main():
    # Argomenti
    parser = argparse.ArgumentParser(description="YOLOv8 TensorFlow Lite Ottimizzato")
    parser.add_argument('--model', required=True, help='Path al modello .tflite')
    parser.add_argument('--camera_id', type=int, default=0, help='ID Camera')
    parser.add_argument('--imgsz', type=int, default=640, help='Dimensione input')
    parser.add_argument('--num_threads', type=int, default=4, help='Thread CPU')
    parser.add_argument('--skip_frames', type=int, default=2, help='Frame da saltare (0=no skip)')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='Soglia confidence')
    parser.add_argument('--iou_threshold', type=float, default=0.45, help='Soglia IoU per NMS')
    parser.add_argument('--resolution', type=str, default='640x480', help='Risoluzione camera (WxH)')
    parser.add_argument('--show_detailed_stats', action='store_true', help='Mostra statistiche dettagliate')
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
    
    # Inizializza camera con settings ottimizzati
    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer minimo per ridurre latenza
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Codec efficiente
    
    # Verifica impostazioni camera
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📷 Camera configurata: {actual_width}x{actual_height} @ {actual_fps}fps")
    
    # Contatori performance
    fps_counter = FPSCounter()
    frame_count = 0
    detection_count = 0
    last_detections = []
    
    # Buffer per statistiche dettagliate
    timing_stats = {
        'inference': deque(maxlen=100),
        'preprocess': deque(maxlen=100),
        'postprocess': deque(maxlen=100),
        'nms': deque(maxlen=100),
        'total': deque(maxlen=100)
    }
    
    print(f" Avvio detection ottimizzata")
    print("Controlli: 'q'=esci, 's'=statistiche, 'd'=dettagli")
    start_global = time.time()
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(" Errore lettura frame")
                break
            
            frame_count += 1
            
            # Logica skip frame migliorata
            should_detect = (args.skip_frames == 0) or (frame_count % (args.skip_frames + 1) == 0)
            
            if should_detect:
                # Esegui detection
                detections, timing_info = detector.detect(frame)
                last_detections = detections
                detection_count += 1
                
                # Aggiorna statistiche timing
                for key, value in timing_info.items():
                    timing_stats[key].append(value)
            else:
                # Usa ultima detection
                detections = last_detections
            
            # Drawing
            detector.draw_detections(frame, detections)
            
            # Aggiorna FPS
            current_fps = fps_counter.update()
            avg_fps = fps_counter.get_average()
            
            # Info overlay ottimizzato
            info_y = 25
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if timing_stats['inference']:
                avg_inference = np.mean(timing_stats['inference']) * 1000
                cv2.putText(frame, f"Inference: {avg_inference:.0f}ms", (10, info_y + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.putText(frame, f"Objects: {len(detections)}", (10, info_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.putText(frame, f"Detections: {detection_count}", (10, info_y + 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow("YOLOv8 Ottimizzato", frame)
            
            # Input handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print(f"\n Statistiche Generali:")
                print(f"   FPS medio: {avg_fps:.2f}")
                print(f"   Frame totali: {frame_count}")
                print(f"   Detection eseguite: {detection_count}")
                if timing_stats['inference']:
                    print(f"   Inference media: {np.mean(timing_stats['inference'])*1000:.1f}ms")
            elif key == ord('d') and args.show_detailed_stats:
                print(f"\n Statistiche Dettagliate:")
                for key, values in timing_stats.items():
                    if values:
                        avg_time = np.mean(values) * 1000
                        print(f"   {key.capitalize()}: {avg_time:.2f}ms")
    
    except KeyboardInterrupt:
        print("\n⏹ Interruzione utente")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        total_duration = time.time() - start_global
        real_fps = frame_count / total_duration
        print(f"   ?? FPS reali (compresi frame saltati e latenza): {real_fps:.2f}")
        # Statistiche finali
        final_fps = fps_counter.get_average()
        print(f"\n Performance finale:")
        print(f"   FPS medio: {final_fps:.2f}")
        print(f"   Frame totali: {frame_count}")
        print(f"   Detection eseguite: {detection_count}")
        print(f"   Rapporto detection/frame: {detection_count/frame_count:.2f}")
        
        if timing_stats['inference']:
            print(f"   Tempi medi:")
            for key, values in timing_stats.items():
                if values:
                    avg_time = np.mean(values) * 1000
                    print(f"     {key.capitalize()}: {avg_time:.2f}ms")

        print(" Rilascio risorse completato")

if __name__ == "__main__":
    
    main()
