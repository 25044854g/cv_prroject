import cv2
import numpy as np
from ultralytics import YOLO

print("Loading models...")
yolo_detect = YOLO('yolov8m.pt')  # Object detection
yolo_hand = YOLO('yolov8n-hand.pt')  # Hand detection
print("Models loaded\n")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

print("Hand-Object Detection Started")
print("Point your hand toward an object")
print("Press 'q' to exit...\n")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    height, width, c = frame.shape
    
    # 1. Object detection
    results_detect = yolo_detect(frame)
    annotated_frame = results_detect[0].plot()
    
    # 2. Hand detection
    results_hand = yolo_hand(frame)
    
    # Get hand position (center of bounding box)
    hand_x, hand_y = None, None
    if results_hand[0].boxes is not None and len(results_hand[0].boxes) > 0:
        # Get first hand detected
        hand_box = results_hand[0].boxes[0]
        x1, y1, x2, y2 = hand_box.xyxy[0]
        hand_x = int((x1 + x2) / 2)
        hand_y = int((y1 + y2) / 2)
        
        # Draw hand bounding box
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        # Draw hand center point
        cv2.circle(annotated_frame, (hand_x, hand_y), 12, (0, 0, 255), -1)
        cv2.putText(annotated_frame, "HAND", (hand_x - 40, hand_y - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        if frame_count % 30 == 0:
            print(f"Hand detected at: ({hand_x}, {hand_y})")
    
    # 3. Analyze hand-object relationship
    boxes = results_detect[0].boxes
    if hand_x is not None and len(boxes) > 0:
        # Get first detected object
        box = boxes[0]
        
        x1, y1, x2, y2 = box.xyxy[0]
        obj_x = int((x1 + x2) / 2)
        obj_y = int((y1 + y2) / 2)
        
        class_id = int(box.cls[0])
        class_name = yolo_detect.names[class_id]
        confidence = float(box.conf[0])
        
        # Calculate distance
        dx = hand_x - obj_x
        dy = hand_y - obj_y
        distance = int(np.sqrt(dx**2 + dy**2))
        
        # Draw line between hand and object
        cv2.line(annotated_frame, (hand_x, hand_y), (obj_x, obj_y), (255, 255, 0), 3)
        
        # Display object info
        cv2.putText(annotated_frame, f"Object: {class_name} ({confidence:.1%})", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(annotated_frame, f"Distance: {distance} px", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Determine direction
        threshold = 80
        direction = ""
        instruction = ""
        
        if abs(dx) > abs(dy):
            if dx < -threshold:
                direction = "Hand RIGHT of object"
                instruction = "Move LEFT"
            elif dx > threshold:
                direction = "Hand LEFT of object"
                instruction = "Move RIGHT"
            else:
                direction = "Aligned horizontally"
        else:
            if dy < -threshold:
                direction = "Hand BELOW object"
                instruction = "Move UP"
            elif dy > threshold:
                direction = "Hand ABOVE object"
                instruction = "Move DOWN"
            else:
                direction = "Aligned vertically"
        
        cv2.putText(annotated_frame, direction, 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        cv2.putText(annotated_frame, instruction, 
                   (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
        
        # Alert when close
        if distance < 120:
            cv2.putText(annotated_frame, "READY TO GRAB!", 
                       (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.rectangle(annotated_frame, (5, 185), (350, 220), (0, 255, 0), 3)
    else:
        if hand_x is None:
            cv2.putText(annotated_frame, "No hand detected", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if len(boxes) == 0:
            cv2.putText(annotated_frame, "No object detected", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Display
    if frame_count % 2 == 0:
        cv2.imshow('Hand-Object Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exited")
        break

cap.release()
cv2.destroyAllWindows()