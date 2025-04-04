import cv2
import numpy as np
from utils.alerts import play_alert
import time

# Configuration
ALERT_COOLDOWN = 5  # seconds between alerts
CONFIDENCE_THRESHOLD = 0.9  # Only consider detections above 70% confidence

# PPE items with their minimum detection thresholds
REQUIRED_PPE = {
    "helmet": {"min_confidence": 0.7, "present": False, "color": (0, 255, 0)},
    "vest": {"min_confidence": 0.65, "present": False, "color": (0, 255, 0)},
    "gloves": {"min_confidence": 0.6, "present": False, "color": (0, 255, 0)},
    "boots": {"min_confidence": 0.6, "present": False, "color": (0, 255, 0)}
}

def reset_ppe_status():
    """Reset detection status for all PPE items"""
    for item in REQUIRED_PPE:
        REQUIRED_PPE[item]["present"] = False

def detect_ppe(model, frame):
    reset_ppe_status()  # Clear previous frame's detections
    missing_ppe = []
    
    # Process detections with higher confidence threshold
    results = model(frame, conf=0.9)  # 80% confidence threshold
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confidences):
            label = model.names[int(cls)].lower().strip()
            
            # Only process required PPE items
            if label in REQUIRED_PPE and conf > REQUIRED_PPE[label]["min_confidence"]:
                REQUIRED_PPE[label]["present"] = True
                color = REQUIRED_PPE[label]["color"]
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, 
                           f"{label}: {conf:.2f}", 
                           (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, color, 2)
    
    # Determine missing PPE
    missing_ppe = [item for item in REQUIRED_PPE if not REQUIRED_PPE[item]["present"]]
    
    # Improved alert generation with confidence check
    if missing_ppe:
        # Check if any missing item has low confidence
        low_confidence_items = [
            item for item in missing_ppe 
            if REQUIRED_PPE[item]["min_confidence"] < 0.6
        ]
        
        if low_confidence_items:
            alert_message = f"Warning! Possibly missing: {', '.join(low_confidence_items)} (Low confidence)"
        else:
            alert_message = f"Safety violation! Confirmed missing: {', '.join(missing_ppe)}"
        
        play_alert(alert_message)
    
    return frame, missing_ppe