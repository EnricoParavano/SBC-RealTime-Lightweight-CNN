"""Terminal-based object detection script for efficientDet Lite models."""
import argparse
import sys
import time
import threading
from queue import Queue
from collections import deque

import cv2
import numpy as np
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision


class TerminalObjectDetector:
    def __init__(self, model_path: str, num_threads: int, score_threshold: float = 0.3, 
                 max_results: int = 10, input_width: int = None, input_height: int = None):
        """Initialize the terminal object detector."""
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.latest_result = None
        self.running = False
        
        # Image resizing parameters
        self.input_width = input_width
        self.input_height = input_height
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps_history = deque(maxlen=30)
        self.frame_count = 0
        
        # Initialize detector
        base_options = core.BaseOptions(
            file_name=model_path, 
            num_threads=num_threads
        )
        detection_options = processor.DetectionOptions(
            max_results=max_results, 
            score_threshold=score_threshold
        )
        options = vision.ObjectDetectorOptions(
            base_options=base_options, 
            detection_options=detection_options
        )
        self.detector = vision.ObjectDetector.create_from_options(options)
        
        # Start inference thread
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        
    def _inference_worker(self):
        """Worker thread for running inference."""
        while self.running:
            try:
                if not self.frame_queue.empty():
                    rgb_image = self.frame_queue.get_nowait()
                    
                    # Create tensor and run detection
                    input_tensor = vision.TensorImage.create_from_array(rgb_image)
                    detection_result = self.detector.detect(input_tensor)
                    
                    # Store result
                    if self.result_queue.full():
                        try:
                            self.result_queue.get_nowait()
                        except:
                            pass
                    self.result_queue.put(detection_result)
                    
                    # Clear remaining frames
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            break
                            
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Inference error: {e}")
                continue
                
    def start(self):
        """Start the detector."""
        self.running = True
        self.inference_thread.start()
        
    def stop(self):
        """Stop the detector."""
        self.running = False
        if self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.0)
            
    def process_frame(self, image: np.ndarray) -> dict:
        """Process a single frame and return detection results."""
        # Resize image if dimensions are specified
        if self.input_width and self.input_height:
            image = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB for inference
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Add frame to queue for inference
        if not self.frame_queue.full():
            self.frame_queue.put(rgb_image)
        
        # Get latest detection result
        result_updated = False
        while not self.result_queue.empty():
            try:
                self.latest_result = self.result_queue.get_nowait()
                result_updated = True
            except:
                break
        
        self.frame_count += 1
        
        return {
            'detections': self.latest_result,
            'result_updated': result_updated,
            'frame_count': self.frame_count
        }
    
    def get_fps(self) -> float:
        """Calculate and return current FPS."""
        self.fps_counter += 1
        
        if self.fps_counter % 10 == 0:
            current_time = time.time()
            fps = 10 / (current_time - self.fps_start_time)
            self.fps_history.append(fps)
            self.fps_start_time = current_time
            
        return sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0


def print_detections(detections, frame_count: int, fps: float):
    """Print detection results to terminal."""
    print(f"\n--- Frame {frame_count} | FPS: {fps:.1f} ---")
    
    if not detections or not detections.detections:
        print("No objects detected")
        return
    
    print(f"Detected {len(detections.detections)} object(s):")
    
    for i, detection in enumerate(detections.detections):
        # Get category name and score
        category = detection.categories[0]
        category_name = category.category_name if category.category_name else f"ID_{category.index}"
        score = category.score
        
        # Get bounding box
        bbox = detection.bounding_box
        x = int(bbox.origin_x)
        y = int(bbox.origin_y)
        width = int(bbox.width)
        height = int(bbox.height)
        
        print(f"  {i+1}. {category_name}: {score:.2f} confidence")
        print(f"     Position: x={x}, y={y}, width={width}, height={height}")


def find_available_cameras():
    """Find all available camera indices."""
    available_cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras


def run_terminal(model: str, camera_id: int, width: int, height: int, num_threads: int,
                score_threshold: float = 0.3, max_frames: int = None, 
                input_width: int = None, input_height: int = None) -> None:
    """Run object detection in terminal mode."""
    
    # Find available cameras
    available_cameras = find_available_cameras()
    print(f"Available cameras: {available_cameras}")
    
    if not available_cameras:
        sys.exit('ERROR: No cameras found')
    
    if camera_id not in available_cameras:
        print(f"Camera {camera_id} not available. Using camera {available_cameras[0]} instead.")
        camera_id = available_cameras[0]
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        sys.exit(f'ERROR: Unable to open camera {camera_id}')
    
    # Verify camera settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\nCamera {camera_id} initialized:")
    print(f"Camera resolution: {actual_width}x{actual_height}")
    print(f"Camera FPS: {actual_fps}")
    if input_width and input_height:
        print(f"Model input size: {input_width}x{input_height}")
    else:
        print(f"Model input size: Same as camera ({actual_width}x{actual_height})")
    print(f"Score threshold: {score_threshold}")
    print(f"CPU threads: {num_threads}")
    
    # Initialize detector
    detector = TerminalObjectDetector(
        model, num_threads, score_threshold, input_width=input_width, input_height=input_height
    )
    detector.start()
    
    print("\n" + "="*50)
    print("OBJECT DETECTION STARTED")
    print("Press Ctrl+C to stop")
    print("="*50)
    
    frames_processed = 0
    
    try:
        while True:
            success, image = cap.read()
            if not success:
                print("Failed to read frame")
                continue
            
            # Process frame
            result = detector.process_frame(image)
            
            # Print results only when new detections are available
            if result['result_updated']:
                fps = detector.get_fps()
                print_detections(result['detections'], result['frame_count'], fps)
            
            frames_processed += 1
            
            # Stop after max_frames if specified
            if max_frames and frames_processed >= max_frames:
                print(f"\nProcessed {max_frames} frames. Stopping...")
                break
                
            # Small delay to avoid overwhelming the terminal
            time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n\nStopping detection...")
        
    finally:
        detector.stop()
        cap.release()
        print(f"Total frames processed: {frames_processed}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Terminal-based real-time object detection")
    
    parser.add_argument(
        '--model',
        help='Path of the object detection model.',
        required=False,
        default='efficientdet_lite0.tflite')
    parser.add_argument(
        '--cameraId', 
        help='Id of camera.', 
        required=False, 
        type=int, 
        default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        type=int,
        default=640)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        type=int,
        default=480)
    parser.add_argument(
        '--numThreads',
        help='Number of CPU threads to run the model.',
        required=False,
        type=int,
        default=4)
    parser.add_argument(
        '--scoreThreshold',
        help='Minimum confidence score for detections.',
        required=False,
        type=float,
        default=0.3)
    parser.add_argument(
        '--maxFrames',
        help='Maximum number of frames to process (optional).',
        required=False,
        type=int,
        default=None)
    parser.add_argument(
        '--inputWidth',
        help='Width to resize input images for the model (optional).',
        required=False,
        type=int,
        default=None)
    parser.add_argument(
        '--inputHeight',
        help='Height to resize input images for the model (optional).',
        required=False,
        type=int,
        default=None)
    
    args = parser.parse_args()
    
    run_terminal(args.model, args.cameraId, args.frameWidth, args.frameHeight,
                args.numThreads, args.scoreThreshold, args.maxFrames, 
                args.inputWidth, args.inputHeight)


if __name__ == '__main__':
    main()
