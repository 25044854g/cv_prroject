import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

print("✓ MediaPipe version:", mp.__version__)

# 获取路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

# 初始化 YOLO
yolo_model = YOLO(os.path.join(project_dir, 'yolov8m.pt'))

# 初始化 MediaPipe 手部检测
model_path = os.path.join(project_dir, 'hand_landmarker.task')

if not os.path.exists(model_path):
    print(f"ERROR: Cannot find {model_path} file")
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

print("✓ Detection Started")
print("Press 'q' to exit...\n")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    height, width, c = frame.shape
    
    # 创建黑色背景
    annotated_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 1. 鼠标检测
    results_yolo = yolo_model(frame)
    boxes = results_yolo[0].boxes
    
    mouse_detected = False
    mouse_x, mouse_y = None, None
    
    for box in boxes:
        class_id = int(box.cls[0])
        class_name = yolo_model.names[class_id]
        
        if class_name.lower() == "mouse":
            mouse_detected = True
            x1, y1, x2, y2 = box.xyxy[0]
            mouse_x = (int(x1) + int(x2)) // 2
            mouse_y = (int(y1) + int(y2)) // 2
            
            # 绘制鼠标框（蓝色）
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                         (255, 0, 0), 3)
            cv2.circle(annotated_frame, (mouse_x, mouse_y), 8, (255, 0, 0), -1)
            break
    
    # 2. 手部检测
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    
    hand_detected = False
    hand_x, hand_y = None, None
    
    if detection_result.hand_landmarks:
        hand_detected = True
        for hand_landmarks in detection_result.hand_landmarks:
            middle_finger = hand_landmarks[12]
            hand_x = int(middle_finger.x * width)
            hand_y = int(middle_finger.y * height)
            
            # 绘制手部骨架
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17)
            ]
            
            for landmark in hand_landmarks:
                lx = int(landmark.x * width)
                ly = int(landmark.y * height)
                cv2.circle(annotated_frame, (lx, ly), 4, (0, 255, 0), -1)
            
            for start, end in connections:
                start_point = (int(hand_landmarks[start].x * width), 
                              int(hand_landmarks[start].y * height))
                end_point = (int(hand_landmarks[end].x * width), 
                            int(hand_landmarks[end].y * height))
                cv2.line(annotated_frame, start_point, end_point, (0, 255, 0), 2)
            
            # 绘制中指尖（红色大圆）
            cv2.circle(annotated_frame, (hand_x, hand_y), 12, (0, 0, 255), -1)
    
    # 3. 显示检测状态
    cv2.putText(annotated_frame, "=== Detection Status ===", 
               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 手部检测状态
    hand_status = "✓ Hand Detected" if hand_detected else "✗ No Hand"
    hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    cv2.putText(annotated_frame, hand_status, 
               (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, hand_color, 2)
    
    # 鼠标检测状态
    mouse_status = "✓ Mouse Detected" if mouse_detected else "✗ No Mouse"
    mouse_color = (0, 255, 0) if mouse_detected else (0, 0, 255)
    cv2.putText(annotated_frame, mouse_status, 
               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, mouse_color, 2)
    
    # 显示坐标
    if hand_detected:
        cv2.putText(annotated_frame, f"Hand: ({hand_x}, {hand_y})", 
                   (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
    
    if mouse_detected:
        cv2.putText(annotated_frame, f"Mouse: ({mouse_x}, {mouse_y})", 
                   (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
    
    # 显示
    if frame_count % 3 == 0:
        cv2.imshow('Step 1: Hand & Mouse Detection', annotated_frame)
    
    if cv2.waitKey(500) & 0xFF == ord('q'):
        print("Exited")
        break

cap.release()
cv2.destroyAllWindows()