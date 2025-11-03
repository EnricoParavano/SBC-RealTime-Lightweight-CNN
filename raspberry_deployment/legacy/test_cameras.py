#!/usr/bin/env python3
"""Script to test and list available cameras on the system."""

import cv2
import sys

def test_camera_detailed(camera_id):
    """Test a specific camera and return detailed info."""
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        return None
    
    # Get camera properties
    info = {
        'id': camera_id,
        'backend': cap.get(cv2.CAP_PROP_BACKEND),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'format': cap.get(cv2.CAP_PROP_FORMAT),
        'fourcc': cap.get(cv2.CAP_PROP_FOURCC),
        'brightness': cap.get(cv2.CAP_PROP_BRIGHTNESS),
        'contrast': cap.get(cv2.CAP_PROP_CONTRAST),
        'saturation': cap.get(cv2.CAP_PROP_SATURATION),
        'hue': cap.get(cv2.CAP_PROP_HUE),
        'gain': cap.get(cv2.CAP_PROP_GAIN),
        'exposure': cap.get(cv2.CAP_PROP_EXPOSURE),
    }
    
    # Test if we can actually read frames
    ret, frame = cap.read()
    info['can_read_frames'] = ret
    if ret:
        info['actual_width'] = frame.shape[1]
        info['actual_height'] = frame.shape[0]
        info['channels'] = frame.shape[2] if len(frame.shape) == 3 else 1
    
    cap.release()
    return info

def find_all_cameras():
    """Find all available cameras."""
    available_cameras = []
    
    print("Scanning for cameras...")
    print("=" * 50)
    
    for i in range(10):  # Check first 10 indices
        print(f"Testing camera {i}...", end=" ")
        
        info = test_camera_detailed(i)
        if info:
            available_cameras.append(info)
            print("✓ Found")
        else:
            print("✗ Not available")
    
    return available_cameras

def print_camera_info(camera_info):
    """Print detailed camera information."""
    print(f"\n📹 Camera {camera_info['id']}:")
    print(f"   Backend: {camera_info['backend']}")
    print(f"   Resolution: {camera_info['width']}x{camera_info['height']}")
    print(f"   FPS: {camera_info['fps']}")
    print(f"   Can read frames: {'Yes' if camera_info['can_read_frames'] else 'No'}")
    
    if camera_info['can_read_frames']:
        print(f"   Actual frame size: {camera_info['actual_width']}x{camera_info['actual_height']}")
        print(f"   Channels: {camera_info['channels']}")
    
    print(f"   Format: {camera_info['format']}")
    print(f"   FOURCC: {camera_info['fourcc']}")
    
    # Only print non-zero properties
    properties = ['brightness', 'contrast', 'saturation', 'hue', 'gain', 'exposure']
    non_zero_props = {prop: camera_info[prop] for prop in properties if camera_info[prop] != -1}
    if non_zero_props:
        print("   Properties:", non_zero_props)

def test_specific_camera(camera_id):
    """Test a specific camera with live preview."""
    print(f"Testing camera {camera_id} with live preview...")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Cannot open camera {camera_id}")
        return False
    
    print("Camera opened successfully!")
    print("Press 'q' to quit, 's' to save a test image, 'i' for info")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame")
            break
        
        frame_count += 1
        
        # Add frame counter to image
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Camera: {camera_id}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(f'Camera {camera_id} Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f'camera_{camera_id}_test.jpg'
            cv2.imwrite(filename, frame)
            print(f"📸 Image saved as {filename}")
        elif key == ord('i'):
            print(f"Frame {frame_count}: {frame.shape}")
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def main():
    if len(sys.argv) > 1:
        # Test specific camera
        try:
            camera_id = int(sys.argv[1])
            test_specific_camera(camera_id)
        except ValueError:
            print("Please provide a valid camera ID (integer)")
    else:
        # Find all cameras
        cameras = find_all_cameras()
        
        if not cameras:
            print("\nNo cameras found!")
            print("\nTroubleshooting tips:")
            print("1. Check if camera is connected properly")
            print("2. Make sure camera is not being used by another application")
            print("3. Try running: lsusb (for USB cameras)")
            print("4. Try running: v4l2-ctl --list-devices (for V4L2 devices)")
            return
        
        print(f"\nFound {len(cameras)} camera(s):")
        for camera in cameras:
            print_camera_info(camera)
        
        print(f"\nRecommended camera IDs: {[cam['id'] for cam in cameras if cam['can_read_frames']]}")
        print("\nTo test a specific camera, run:")
        print("python test_cameras.py <camera_id>")

if __name__ == "__main__":
    main()