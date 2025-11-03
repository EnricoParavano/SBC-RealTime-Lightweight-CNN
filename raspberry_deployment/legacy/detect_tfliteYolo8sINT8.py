import cv2
import numpy as np
import time
import argparse
import tflite_runtime.interpreter as tflite
from collections import deque
from utils import vectorized_nms

class OptimizedYOLODetector:
    def __init__(self, model_path, num_threads=4, conf_threshold=1e-6, 
                 iou_threshold=0.45, min_area=100):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_area = min_area
        
        self.classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus',
            7: 'truck', 15: 'cat', 16: 'dog', 39: 'bottle', 56: 'chair',
            57: 'couch', 62: 'tv', 63: 'laptop', 67: 'cell phone'
        }
        
        self.colors = {
            0: (255, 100, 100),
            2: (100, 100, 255), 3: (100, 100, 255), 5: (100, 100, 255), 7: (100, 100, 255),
        }
        self.default_color = (100, 255, 100)

        print("🔄 Loading INT8 model...")
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        self.input_tensor = np.zeros(self.input_shape, dtype=np.int8)
        print(f"Model loaded: shape={self.input_shape}")

    def preprocess_optimized(self, frame, target_size):
        resized = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.int8) / 255.0
        quant = np.clip(np.round(rgb / self.input_details[0]['quantization'][0]) + self.input_details[0]['quantization'][1], -128, 127).astype(np.int8)
        self.input_tensor[0] = quant
        return self.input_tensor

    def postprocess_optimized(self, output_data, frame_shape):
        dequant = (output_data.astype(np.int8) - self.output_details[0]['quantization'][1]) * self.output_details[0]['quantization'][0]
        predictions = dequant.transpose(1, 0)
        coords = predictions[:, :4]
        class_confs = predictions[:, 4:]
        max_conf_indices = np.argmax(class_confs, axis=1)
        max_confs = class_confs[np.arange(len(class_confs)), max_conf_indices]
        valid_mask = max_confs > self.conf_threshold
        if not np.any(valid_mask):
            return [], [], []
        valid_coords = coords[valid_mask]
        valid_confs = max_confs[valid_mask]
        valid_classes = max_conf_indices[valid_mask]
        h, w = frame_shape[:2]
        cx, cy, bw, bh = valid_coords.T
        x1 = ((cx - bw/2) * w).astype(np.int32)
        y1 = ((cy - bh/2) * h).astype(np.int32)
        x2 = ((cx + bw/2) * w).astype(np.int32)
        y2 = ((cy + bh/2) * h).astype(np.int32)
        x1 = np.clip(x1, 0, w-1)
        y1 = np.clip(y1, 0, h-1)
        x2 = np.clip(x2, 0, w-1)
        y2 = np.clip(y2, 0, h-1)
        areas = (x2 - x1) * (y2 - y1)
        area_mask = (areas > self.min_area) & (x2 > x1) & (y2 > y1)
        if not np.any(area_mask):
            return [], [], []
        final_boxes = np.column_stack([x1, y1, x2, y2])[area_mask]
        final_scores = valid_confs[area_mask]
        final_classes = valid_classes[area_mask]
        keep = vectorized_nms(final_boxes, final_scores, self.iou_threshold)
        return final_boxes[keep].tolist(), final_scores[keep].tolist(), final_classes[keep].tolist()

    def detect(self, frame):
        input_size = self.input_shape[1]
        tensor = self.preprocess_optimized(frame, input_size)
        self.interpreter.set_tensor(self.input_details[0]['index'], tensor)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return self.postprocess_optimized(output_data, frame.shape)

    def draw_detections(self, frame, boxes, scores, classes):
        for box, score, cls in zip(boxes, scores, classes):
            color = self.colors.get(cls, self.default_color)
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{self.classes.get(cls, cls)}: {score:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to .tflite model')
    parser.add_argument('--camera_id', type=int, default=0)
    parser.add_argument('--resolution', type=str, default='640x480', help='WxH')
    parser.add_argument('--num_threads', type=int, default=4)
    parser.add_argument('--conf_threshold', type=float, default=1e-6)
    parser.add_argument('--iou_threshold', type=float, default=0.45)
    args = parser.parse_args()

    width, height = map(int, args.resolution.split('x'))
    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    detector = OptimizedYOLODetector(
        model_path=args.model,
        num_threads=args.num_threads,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturing frame")
            break

        boxes, scores, classes = detector.detect(frame)
        detector.draw_detections(frame, boxes, scores, classes)

        cv2.imshow('Detections', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
