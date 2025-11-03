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

def preprocess_frame(frame, input_shape, input_scale, input_zero_point):
    resized = cv2.resize(frame, (input_shape[1], input_shape[2]), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    quant = np.clip(np.round(rgb / input_scale) + input_zero_point, -128, 127).astype(np.int8)
    return quant

def postprocess_output(output_data, frame_shape, conf_threshold, iou_threshold, min_area=100):
    dequant = (output_data.astype(np.float32) - output_data[0]['quantization'][1]) * output_data[0]['quantization'][0]
    predictions = dequant.transpose(1, 0)
    coords = predictions[:, :4]
    class_confs = predictions[:, 4:]
    
    max_conf_indices = np.argmax(class_confs, axis=1)
    max_confs = class_confs[np.arange(len(class_confs)), max_conf_indices]
    
    valid_mask = max_confs > conf_threshold
    if not np.any(valid_mask):
        return [], [], []
    
    valid_coords = coords[valid_mask]
    valid_confs = max_confs[valid_mask]
    valid_classes = max_conf_indices[valid_mask]
    
    h, w = frame_shape[:2]
    cx, cy, bw, bh = valid_coords.T
    
    x1 = ((cx - bw / 2) * w).astype(np.int32)
    y1 = ((cy - bh / 2) * h).astype(np.int32)
    x2 = ((cx + bw / 2) * w).astype(np.int32)
    y2 = ((cy + bh / 2) * h).astype(np.int32)
    
    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)
    x2 = np.clip(x2, 0, w - 1)
    y2 = np.clip(y2, 0, h - 1)
    
    areas = (x2 - x1) * (y2 - y1)
    area_mask = (areas > min_area) & (x2 > x1) & (y2 > y1)
    
    if not np.any(area_mask):
        return [], [], []
    
    final_boxes = np.column_stack([x1, y1, x2, y2])[area_mask]
    final_scores = valid_confs[area_mask]
    final_classes = valid_classes[area_mask]
    
    keep = vectorized_nms(final_boxes, final_scores, iou_threshold)
    return final_boxes[keep].tolist(), final_scores[keep].tolist(), final_classes[keep].tolist()