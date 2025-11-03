"""Optimized utility functions for object detection visualization."""
import cv2
import numpy as np
from tflite_support.task import processor

# Optimized constants
_MARGIN = 10
_ROW_SIZE = 10
_FONT_SIZE = 0.7  # Reduced for better performance
_FONT_THICKNESS = 2
_TEXT_COLOR = (0, 255, 0)  # Green for better visibility
_BBOX_COLOR = (0, 255, 0)  # Green bounding boxes
_BBOX_THICKNESS = 2

# Pre-defined colors for different classes (more efficient than calculating each time)
_COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
    (255, 192, 203), # Pink
    (128, 128, 128), # Gray
]

def visualize(image: np.ndarray, detection_result: processor.DetectionResult) -> np.ndarray:
    """Optimized visualization of detection results.
    
    Args:
        image: The input RGB image.
        detection_result: The list of all "Detection" entities to visualize.
    
    Returns:
        Image with bounding boxes and labels.
    """
    if not detection_result.detections:
        return image
    
    # Get image dimensions for optimization checks
    img_height, img_width = image.shape[:2]
    
    for i, detection in enumerate(detection_result.detections):
        # Get bounding box coordinates
        bbox = detection.bounding_box
        x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
        x2, y2 = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))
        
        # Skip if bounding box is too small
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            continue
        
        # Use color cycling for different detections
        color = _COLORS[i % len(_COLORS)]
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, _BBOX_THICKNESS)
        
        # Get category information
        if detection.categories:
            category = detection.categories[0]
            category_name = category.category_name
            confidence = category.score
            
            # Create optimized label text
            label = f"{category_name}: {confidence:.2f}"
            
            # Calculate text size for background rectangle
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, _FONT_SIZE, _FONT_THICKNESS
            )
            
            # Calculate label position
            label_x = x1 + _MARGIN
            label_y = y1 - _MARGIN if y1 - _MARGIN > text_height else y1 + text_height + _MARGIN
            
            # Ensure label is within image bounds
            label_x = max(0, min(label_x, img_width - text_width))
            label_y = max(text_height, min(label_y, img_height - 5))
            
            # Draw background rectangle for text (optional, for better readability)
            cv2.rectangle(
                image,
                (label_x - 2, label_y - text_height - 2),
                (label_x + text_width + 2, label_y + 2),
                (0, 0, 0),  # Black background
                -1  # Filled rectangle
            )
            
            # Draw text
            cv2.putText(
                image, 
                label, 
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                _FONT_SIZE, 
                color, 
                _FONT_THICKNESS,
                cv2.LINE_AA  # Anti-aliased text for better quality
            )
    
    return image


def visualize_minimal(image: np.ndarray, detection_result: processor.DetectionResult) -> np.ndarray:
    """Ultra-fast minimal visualization - only bounding boxes.
    
    Use this for maximum performance when you don't need labels.
    """
    if not detection_result.detections:
        return image
    
    img_height, img_width = image.shape[:2]
    
    for i, detection in enumerate(detection_result.detections):
        bbox = detection.bounding_box
        x1 = max(0, min(int(bbox.origin_x), img_width - 1))
        y1 = max(0, min(int(bbox.origin_y), img_height - 1))
        x2 = max(0, min(int(bbox.origin_x + bbox.width), img_width - 1))
        y2 = max(0, min(int(bbox.origin_y + bbox.height), img_height - 1))
        
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            color = _COLORS[i % len(_COLORS)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    return image


def get_detection_info(detection_result: processor.DetectionResult) -> list:
    """Extract detection information without visualization.
    
    Returns:
        List of dictionaries containing detection information.
    """
    detections_info = []
    
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        category = detection.categories[0] if detection.categories else None
        
        detection_info = {
            'bbox': {
                'x': int(bbox.origin_x),
                'y': int(bbox.origin_y),
                'width': int(bbox.width),
                'height': int(bbox.height)
            },
            'category': category.category_name if category else 'unknown',
            'confidence': category.score if category else 0.0
        }
        
        detections_info.append(detection_info)
    
    return detections_info