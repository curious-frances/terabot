import numpy as np

def non_max_suppression_yolov8(boxes, class_count, keypoint_count=0):
    """
    Perform non-maximum suppression on YOLOv8 detection boxes.
    
    Args:
        boxes: List of detection boxes, each containing [x, y, w, h, score, class_idx, ...keypoints]
        class_count: Number of classes in the model
        keypoint_count: Number of keypoints per detection (0 for object detection)
    
    Returns:
        List of filtered boxes
    """
    if len(boxes) == 0:
        return []

    # Convert to numpy array for easier manipulation
    boxes = np.array(boxes)
    
    # Sort boxes by score in descending order
    scores = boxes[:, 4]
    indices = np.argsort(-scores)
    boxes = boxes[indices]
    
    # Initialize list of picked boxes
    picked = []
    
    while len(boxes) > 0:
        # Take the box with highest score
        picked.append(boxes[0])
        
        if len(boxes) == 1:
            break
            
        # Calculate IoU with remaining boxes
        box = boxes[0]
        remaining_boxes = boxes[1:]
        
        # Get coordinates
        x1 = np.maximum(box[0] - box[2]/2, remaining_boxes[:, 0] - remaining_boxes[:, 2]/2)
        y1 = np.maximum(box[1] - box[3]/2, remaining_boxes[:, 1] - remaining_boxes[:, 3]/2)
        x2 = np.minimum(box[0] + box[2]/2, remaining_boxes[:, 0] + remaining_boxes[:, 2]/2)
        y2 = np.minimum(box[1] + box[3]/2, remaining_boxes[:, 1] + remaining_boxes[:, 3]/2)
        
        # Calculate intersection area
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate union area
        box_area = box[2] * box[3]
        remaining_areas = remaining_boxes[:, 2] * remaining_boxes[:, 3]
        union = box_area + remaining_areas - intersection
        
        # Calculate IoU
        iou = intersection / union
        
        # Keep boxes with IoU less than threshold
        boxes = remaining_boxes[iou < 0.5]
    
    return picked 