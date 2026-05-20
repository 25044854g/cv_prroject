import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

print("✓ MediaPipe version:", mp.__version__)

# Resolve paths relative to this script's directory
base_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(base_dir)

# Initialize YOLO with Medium model for better accuracy
yolo_model = YOLO(os.path.join(project_dir, 'yolov8m.pt'))

# Initialize MediaPipe hand detection
model_path = os.path.join(base_dir, 'hand_landmarker.task')

if not os.path.exists(model_path):
    print(f"ERROR: Cannot find {model_path} file")
    print("Please download the model file first")
    exit()

print("Loading MediaPipe model...")
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
print("✓ MediaPipe model loaded\n")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

print("✓ Hand-Object Detection Started (MediaPipe + YOLOv8m)")
print("Point your hand toward an object (NOT a person)")
print("Press 'q' to exit...\n")

frame_count = 0

# COCO class list for reference
PERSON_CLASS_ID = 0  # In COCO dataset, 'person' is class 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    height, width, c = frame.shape
    
    # 1. Object detection (using yolov8m for better accuracy)
    results_yolo = yolo_model(frame)
    annotated_frame = results_yolo[0].plot()
    
    # 2. Hand detection (MediaPipe new API)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    
    # Get hand position
    hand_x, hand_y = None, None
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Get middle finger tip (landmark 12)
            middle_finger = hand_landmarks[12]
            hand_x = int(middle_finger.x * width)
            hand_y = int(middle_finger.y * height)
            
            # Draw all landmarks (green dots)
            for landmark in hand_landmarks:
                lx = int(landmark.x * width)
                ly = int(landmark.y * height)
                cv2.circle(annotated_frame, (lx, ly), 4, (0, 255, 0), -1)
            
            # Draw hand skeleton
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17)
            ]
            
            for start, end in connections:
                start_point = (int(hand_landmarks[start].x * width), 
                              int(hand_landmarks[start].y * height))
                end_point = (int(hand_landmarks[end].x * width), 
                            int(hand_landmarks[end].y * height))
                cv2.line(annotated_frame, start_point, end_point, (255, 0, 0), 2)
            
            # Draw red circle at middle finger tip
            cv2.circle(annotated_frame, (hand_x, hand_y), 12, (0, 0, 255), -1)
            cv2.putText(annotated_frame, "HAND", (hand_x - 50, hand_y - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 3. Analyze hand-object relationship
    # IMPORTANT: Filter out 'person' class - only detect non-person objects
    boxes = results_yolo[0].boxes
    target_box = None
    
    if hand_x is not None and len(boxes) > 0:
        # Find first non-person object
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]
            
            # Skip if it's a person
            if class_id == PERSON_CLASS_ID or class_name.lower() == "person":
                if frame_count % 30 == 0:
                    print(f"Skipping person detection")
                continue
            
            # Found a non-person object
            target_box = box
            if frame_count % 30 == 0:
                print(f"Target object found: {class_name}")
            break
    
    # Process the target object (if found and not a person)
    if hand_x is not None and target_box is not None:
        box = target_box
        
        x1, y1, x2, y2 = box.xyxy[0]
        obj_x = (int(x1) + int(x2)) // 2
        obj_y = (int(y1) + int(y2)) // 2
        
        class_id = int(box.cls[0])
        class_name = yolo_model.names[class_id]
        confidence = float(box.conf[0])
        
        # Draw target box with different color (magenta for non-person targets)
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                     (255, 0, 255), 3)
        
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
        
        # Determine direction - Show how to move TOWARD the object
        threshold = 80
        direction = ""
        instruction = ""
        
        if abs(dx) > abs(dy):
            if dx < -threshold:
                # Hand is to the RIGHT of object, move LEFT to reach it
                direction = "Hand is RIGHT of object"
                instruction = "Move LEFT to object"
                color = (0, 165, 255)  # Orange
            elif dx > threshold:
                # Hand is to the LEFT of object, move RIGHT to reach it
                direction = "Hand is LEFT of object"
                instruction = "Move RIGHT to object"
                color = (0, 165, 255)  # Orange
            else:
                direction = "Aligned horizontally"
                instruction = ""
                color = (0, 255, 0)  # Green
        else:
            if dy > -threshold:
                # Hand is BELOW object, move UP to reach it
                direction = "Hand is BELOW object"
                instruction = "Move UP to object"
                color = (0, 165, 255)  # Orange
            elif dy < threshold:
                # Hand is ABOVE object, move DOWN to reach it
                direction = "Hand is ABOVE object"
                instruction = "Move DOWN to object"
                color = (0, 165, 255)  # Orange
            else:
                direction = "Aligned vertically"
                instruction = ""
                color = (0, 255, 0)  # Green
        
        cv2.putText(annotated_frame, direction, 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(annotated_frame, instruction, 
                   (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        

        
        # Alert when close
        if distance < 120:
            cv2.putText(annotated_frame, "READY TO GRAB!", 
                       (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.rectangle(annotated_frame, (5, 185), (350, 220), (0, 255, 0), 3)
    else:
        if hand_x is None:
            cv2.putText(annotated_frame, "No hand detected", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    
    # Display
    if frame_count % 3 == 0:
        cv2.imshow('Hand-Object Detection (No Person)', annotated_frame)
    
    if cv2.waitKey(500) & 0xFF == ord('q'):
        print("Exited")
        break

cap.release()
cv2.destroyAllWindows()