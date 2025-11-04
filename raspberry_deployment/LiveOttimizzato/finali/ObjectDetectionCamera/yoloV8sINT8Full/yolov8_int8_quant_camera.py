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
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

class OptimizedYOLODetector:
    def __init__(self, model_path, num_threads=4, conf_threshold=0.25,
                 iou_threshold=0.45, min_area=100):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_area = min_area
        
        # COCO classes principali
        self.classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
            25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
            39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
            44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
            49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
            54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
            59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
            64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
            74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
            79: 'toothbrush'
        }
        
        self.colors = {
            0: (255, 100, 100),  # person - rosso
            2: (100, 100, 255),  # car - blu
            3: (100, 100, 255),  # motorcycle - blu
            5: (100, 100, 255),  # bus - blu
            7: (100, 100, 255),  # truck - blu
            15: (255, 200, 100), # cat - arancione
            16: (255, 200, 100), # dog - arancione
        }
        self.default_color = (100, 255, 100)  # verde

        print("?? Caricamento modello quantizzato INT8...")
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Parametri di quantizzazione INPUT
        self.input_scale, self.input_zero_point = self.input_details[0]['quantization']
        print(f"Input quantization: scale={self.input_scale}, zero_point={self.input_zero_point}")
        
        # Parametri di quantizzazione OUTPUT
        self.output_scale, self.output_zero_point = self.output_details[0]['quantization']
        print(f"Output quantization: scale={self.output_scale}, zero_point={self.output_zero_point}")
        
        self.input_shape = self.input_details[0]['shape']
        self.output_shape = self.output_details[0]['shape']
        
        print(f"? Modello caricato:")
        print(f"   Input shape: {self.input_shape}")
        print(f"   Output shape: {self.output_shape}")
        print(f"   Input dtype: {self.input_details[0]['dtype']}")
        print(f"   Output dtype: {self.output_details[0]['dtype']}")
        
        # Pre-allocazione tensori
        self.input_tensor = np.zeros(self.input_shape, dtype=np.int8)

    def preprocess_optimized(self, frame, target_size=640):
        """Preprocessing ottimizzato per input INT8"""
        # Resize mantenendo aspect ratio
        h, w = frame.shape[:2]
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Padding per raggiungere 640x640
        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Conversione BGR -> RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # Quantizzazione: pixel normali (0-255) -> INT8 quantizzato
        if self.input_scale > 0:
            # Normalizzazione standard [0,255] -> [0,1] -> quantizzazione
            normalized = rgb.astype(np.float32) / 255.0
            quantized = np.clip(
                np.round(normalized / self.input_scale) + self.input_zero_point, 
                -128, 127
            ).astype(np.int8)
        else:
            print("Warning: input_scale is 0, using direct conversion")
            quantized = (rgb.astype(np.int32) - 128).astype(np.int8)
        
        self.input_tensor[0] = quantized
        return self.input_tensor, scale, (x_offset, y_offset)

    def postprocess_optimized(self, output_data, frame_shape, scale, offsets):
        """Postprocessing ottimizzato per output INT8"""
        # Dequantizzazione dell'output INT8 -> float32
        if self.output_scale > 0:
            dequant = (output_data.astype(np.float32) - self.output_zero_point) * self.output_scale
        else:
            print("Warning: output_scale is 0, using direct conversion")
            dequant = output_data.astype(np.float32)
        
        # Reshape: da [1, 84, 8400] a [8400, 84]
        predictions = dequant[0].transpose(1, 0)
        
        # Separazione coordinate e confidenze
        coords = predictions[:, :4]  # x_center, y_center, width, height
        class_confs = predictions[:, 4:]  # confidenze per le 80 classi
        
        # Trova classe con confidenza massima per ogni detection
        max_conf_indices = np.argmax(class_confs, axis=1)
        max_confs = class_confs[np.arange(len(class_confs)), max_conf_indices]
        
        # Filtra per confidenza minima
        valid_mask = max_confs > self.conf_threshold
        if not np.any(valid_mask):
            return [], [], []
        
        valid_coords = coords[valid_mask]
        valid_confs = max_confs[valid_mask]
        valid_classes = max_conf_indices[valid_mask]
        
        # Conversione coordinate da formato YOLO a pixel
        h, w = frame_shape[:2]
        x_offset, y_offset = offsets
        
        cx, cy, bw, bh = valid_coords.T
        
        # Correzione per padding e scaling
        cx = (cx * 640 - x_offset) / scale
        cy = (cy * 640 - y_offset) / scale
        bw = bw * 640 / scale
        bh = bh * 640 / scale
        
        # Conversione a coordinate angolari
        x1 = np.clip((cx - bw/2).astype(np.int32), 0, w-1)
        y1 = np.clip((cy - bh/2).astype(np.int32), 0, h-1)
        x2 = np.clip((cx + bw/2).astype(np.int32), 0, w-1)
        y2 = np.clip((cy + bh/2).astype(np.int32), 0, h-1)
        
        # Filtra per area minima
        areas = (x2 - x1) * (y2 - y1)
        area_mask = (areas > self.min_area) & (x2 > x1) & (y2 > y1)
        
        if not np.any(area_mask):
            return [], [], []
        
        final_boxes = np.column_stack([x1, y1, x2, y2])[area_mask]
        final_scores = valid_confs[area_mask]
        final_classes = valid_classes[area_mask]
        
        # NMS
        keep = vectorized_nms(final_boxes, final_scores, self.iou_threshold)
        
        return final_boxes[keep].tolist(), final_scores[keep].tolist(), final_classes[keep].tolist()
        
    def detect(self, frame):
        """Rilevamento oggetti"""
        input_size = 640
        tensor, scale, offsets = self.preprocess_optimized(frame, input_size)
        
        # Inferenza
        self.interpreter.set_tensor(self.input_details[0]['index'], tensor)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return self.postprocess_optimized(output_data, frame.shape, scale, offsets)

    def draw_detections(self, frame, boxes, scores, classes):
        """Disegna le detection sul frame"""
        for box, score, cls in zip(boxes, scores, classes):
            color = self.colors.get(cls, self.default_color)
            x1, y1, x2, y2 = box
            
            # Rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label
            class_name = self.classes.get(cls, f"Class_{cls}")
            label = f"{class_name}: {score:.2f}"
            
            # Background per il testo
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1-label_size[1]-10), 
                         (x1+label_size[0], y1), color, -1)
            
            # Testo
            cv2.putText(frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 INT8 Real-time Detection')
    parser.add_argument('--model', required=True, help='Path to .tflite quantized model')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera ID')
    parser.add_argument('--resolution', type=str, default='640x480', help='Camera resolution WxH')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of CPU threads')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--show_fps', action='store_true', help='Show FPS counter')
    args = parser.parse_args()

    # Setup camera
    width, height = map(int, args.resolution.split('x'))
    cap = cv2.VideoCapture(args.camera_id)
    
    if not cap.isOpened():
        print("? Errore: impossibile aprire la camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Inizializza detector
    try:
        detector = OptimizedYOLODetector(
            model_path=args.model,
            num_threads=args.num_threads,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold
        )
        print("?? Detector inizializzato. Premi 'q' per uscire.")
    except Exception as e:
        print(f"? Errore nel caricamento del modello: {e}")
        return

    # FPS counter
    fps_counter = deque(maxlen=30)
    
    try:
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("? Errore nella cattura del frame")
                break

            # Detection
            boxes, scores, classes = detector.detect(frame)
            
            # Draw detections
            detector.draw_detections(frame, boxes, scores, classes)
            
            # FPS calculation
            end_time = time.time()
            frame_time = end_time - start_time
            fps = 1.0 / frame_time if frame_time > 0 else 0
            fps_counter.append(fps)
            
            if args.show_fps:
                avg_fps = sum(fps_counter) / len(fps_counter)
                cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Info detections
            if boxes:
                cv2.putText(frame, f"Objects: {len(boxes)}", (10, height-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('YOLOv8 INT8 Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n?? Interruzione da tastiera")
    except Exception as e:
        print(f"? Errore durante l'esecuzione: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("? Cleanup completato")

if __name__ == "__main__":
    main()
