import argparse
import sys
import time
import threading
import socket
import struct
from queue import Queue
from collections import deque

import cv2
import numpy as np
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils_optimized


class UDPStreamer:
    """Handle UDP streaming of video frames."""

    def __init__(self, host='127.0.0.1', port=5200, max_packet_size=65507):
        self.host = host
        self.port = port
        self.max_packet_size = max_packet_size
        self.max_chunk_size = max_packet_size - 16  # Leave room for headers

        # Create UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Increase socket buffer size for better performance
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)  # 1MB buffer

        print(f"UDP Streamer initialized - {host}:{port}")

    def send_frame(self, frame):
        """Send frame via UDP with chunking for large frames."""
        try:
            # Encode frame as JPEG with high quality but compressed
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # 80% quality for good compression
            result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)

            if not result:
                print("Failed to encode frame")
                return False

            data = encoded_frame.tobytes()
            data_size = len(data)

            # If data is small enough, send in one packet
            if data_size <= self.max_chunk_size:
                # Header: frame_id(4) + total_chunks(4) + chunk_index(4) + chunk_size(4)
                header = struct.pack('!IIII', int(time.time() * 1000) % 4294967296, 1, 0, data_size)
                packet = header + data
                self.sock.sendto(packet, (self.host, self.port))
            else:
                # Split into chunks
                frame_id = int(time.time() * 1000) % 4294967296
                total_chunks = (data_size + self.max_chunk_size - 1) // self.max_chunk_size

                for i in range(total_chunks):
                    start_idx = i * self.max_chunk_size
                    end_idx = min(start_idx + self.max_chunk_size, data_size)
                    chunk = data[start_idx:end_idx]

                    # Header: frame_id(4) + total_chunks(4) + chunk_index(4) + chunk_size(4)
                    header = struct.pack('!IIII', frame_id, total_chunks, i, len(chunk))
                    packet = header + chunk

                    self.sock.sendto(packet, (self.host, self.port))

                    # Small delay between chunks to avoid overwhelming the network
                    if total_chunks > 1:
                        time.sleep(0.001)

            return True

        except Exception as e:
            print(f"UDP streaming error: {e}")
            return False

    def close(self):
        """Close the UDP socket."""
        self.sock.close()


class OptimizedObjectDetector:
    def __init__(self, model_path: str, num_threads: int, enable_edgetpu: bool,
                 score_threshold: float = 0.3, max_results: int = 3):
        """Initialize the optimized object detector."""
        self.frame_queue = Queue(maxsize=2)  # Small buffer to avoid latency
        self.result_queue = Queue(maxsize=2)
        self.latest_result = None
        self.running = False

        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps_history = deque(maxlen=30)  # Track last 30 FPS measurements

        # Initialize detector
        base_options = core.BaseOptions(
            file_name=model_path,
            use_coral=enable_edgetpu,
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
                # Get frame from queue (non-blocking)
                if not self.frame_queue.empty():
                    rgb_image = self.frame_queue.get_nowait()

                    # Create tensor and run detection
                    input_tensor = vision.TensorImage.create_from_array(rgb_image)
                    detection_result = self.detector.detect(input_tensor)

                    # Store result (overwrite if queue is full)
                    if self.result_queue.full():
                        try:
                            self.result_queue.get_nowait()  # Remove old result
                        except:
                            pass
                    self.result_queue.put(detection_result)

                    # Clear remaining frames to avoid latency buildup
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            break

                time.sleep(0.001)  # Small delay to prevent excessive CPU usage

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

    def process_frame(self, image: np.ndarray) -> np.ndarray:
        """Process a single frame."""
        # Convert BGR to RGB for inference
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Add frame to queue for inference (non-blocking)
        if not self.frame_queue.full():
            self.frame_queue.put(rgb_image)

        # Get latest detection result
        while not self.result_queue.empty():
            try:
                self.latest_result = self.result_queue.get_nowait()
            except:
                break

        # Visualize with latest available result
        if self.latest_result is not None:
            image = utils_optimized.visualize(image, self.latest_result)

        return image

    def get_fps(self) -> float:
        """Calculate and return current FPS."""
        self.fps_counter += 1

        if self.fps_counter % 10 == 0:  # Calculate every 10 frames
            current_time = time.time()
            fps = 10 / (current_time - self.fps_start_time)
            self.fps_history.append(fps)
            self.fps_start_time = current_time

        # Return smoothed FPS
        return sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0


def find_available_cameras():
    """Find all available camera indices."""
    available_cameras = []
    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras


def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool, score_threshold: float = 0.3,
        udp_host: str = '127.0.0.1', udp_port: int = 5200,
        show_local: bool = False) -> None:
    """Run optimized object detection with UDP streaming."""

    # Find available cameras
    available_cameras = find_available_cameras()
    print(f"Available cameras: {available_cameras}")

    if not available_cameras:
        sys.exit('ERROR: No cameras found')

    # If requested camera is not available, use the first available one
    if camera_id not in available_cameras:
        print(f"Camera {camera_id} not available. Using camera {available_cameras[0]} instead.")
        camera_id = available_cameras[0]

    # Initialize camera with optimizations
    cap = cv2.VideoCapture(camera_id)

    # Camera optimizations
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set target FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce latency

    # Additional optimizations for specific backends
    backend = cap.get(cv2.CAP_PROP_BACKEND)
    print(f"Camera backend: {backend}")

    if backend == cv2.CAP_V4L2:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    if not cap.isOpened():
        sys.exit(f'ERROR: Unable to open camera {camera_id}')

    # Verify actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera {camera_id} opened successfully")
    print(f"Requested resolution: {width}x{height}")
    print(f"Actual resolution: {actual_width}x{actual_height}")
    print(f"Camera FPS: {actual_fps}")

    # Initialize UDP streamer
    streamer = UDPStreamer(udp_host, udp_port)

    # Initialize detector
    detector = OptimizedObjectDetector(
        model, num_threads, enable_edgetpu, score_threshold
    )
    detector.start()

    # Display parameters
    font_size = 1
    font_thickness = 1
    text_color = (0, 255, 0)  # Green for better visibility

    print(f"Starting optimized object detection with UDP streaming to {udp_host}:{udp_port}")
    if show_local:
        print("Local display enabled - Press ESC to exit")
    else:
        print("Local display disabled - Press Ctrl+C to exit")

    frame_count = 0
    stream_fps_counter = 0
    stream_fps_start = time.time()

    try:
        while True:
            success, image = cap.read()
            if not success:
                print("Failed to read frame")
                continue

            # Flip image horizontally for mirror effect
            image = cv2.flip(image, 1)

            # Process frame
            image = detector.process_frame(image)

            # Display FPS and streaming info
            fps = detector.get_fps()
            fps_text = f'FPS: {fps:.1f}'
            cv2.putText(image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, text_color, font_thickness)

            # Display resolution info
            res_text = f'Resolution: {actual_width}x{actual_height}'
            cv2.putText(image, res_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, text_color, 1)

            # Display streaming info
            stream_text = f'Streaming to: {udp_host}:{udp_port}'
            cv2.putText(image, stream_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, text_color, 1)

            # Calculate streaming FPS
            stream_fps_counter += 1
            if stream_fps_counter % 30 == 0:  # Calculate every 30 frames
                current_time = time.time()
                streaming_fps = 30 / (current_time - stream_fps_start)
                stream_fps_start = current_time
                print(f"Streaming FPS: {streaming_fps:.1f}")

            # Send frame via UDP
            streamer.send_frame(image)

            # Show local display if enabled
            if show_local:
                cv2.imshow('Object Detection - UDP Streaming', image)

                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break

            frame_count += 1

            # Print status every 100 frames
            if frame_count % 100 == 0:
                print(f"Frames processed: {frame_count}")

    except KeyboardInterrupt:
        print("\nStopping detection and streaming...")

    finally:
        detector.stop()
        streamer.close()
        cap.release()
        if show_local:
            cv2.destroyAllWindows()
        print("Cleanup completed")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
        '--enableEdgeTPU',
        help='Whether to run the model on EdgeTPU.',
        action='store_true',
        required=False,
        default=False)
    parser.add_argument(
        '--scoreThreshold',
        help='Minimum confidence score for detections.',
        required=False,
        type=float,
        default=0.35)
    parser.add_argument(
        '--udpHost',
        help='UDP host address for streaming.',
        required=False,
        type=str,
        default='127.0.0.1')
    parser.add_argument(
        '--udpPort',
        help='UDP port for streaming.',
        required=False,
        type=int,
        default=5200)
    parser.add_argument(
        '--showLocal',
        help='Show local video window.',
        action='store_true',
        required=False,
        default=False)
    args = parser.parse_args()

    print(f"Starting with model: {args.model}")
    print(f"Resolution: {args.frameWidth}x{args.frameHeight}")
    print(f"Threads: {args.numThreads}")
    print(f"EdgeTPU: {args.enableEdgeTPU}")
    print(f"Score threshold: {args.scoreThreshold}")
    print(f"UDP streaming: {args.udpHost}:{args.udpPort}")
    print(f"Local display: {args.showLocal}")

    run(args.model, args.cameraId, args.frameWidth, args.frameHeight,
        args.numThreads, args.enableEdgeTPU, args.scoreThreshold,
        args.udpHost, args.udpPort, args.showLocal)


if __name__ == '__main__':
    main()