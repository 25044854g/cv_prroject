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

# 初始化 YOLO
yolo_model = YOLO(os.path.join(script_dir, 'yolov8m.pt'))

# 初始化 MediaPipe 手部检测
model_path = os.path.join(script_dir, 'hand_landmarker.task')

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

print("✓ Depth Detection Started")
print("Press 'q' to exit...\n")

frame_count = 0


def estimate_hand_depth(hand_landmarks, frame_width, frame_height):
    """
    估计手的深度
    基于手的大小：手越大 = 离摄像头越近
    返回 0-1，1 表示离摄像头最近
    """
    hand_x_coords = [lm.x for lm in hand_landmarks]
    hand_y_coords = [lm.y for lm in hand_landmarks]
    
    hand_x_min = min(hand_x_coords)
    hand_x_max = max(hand_x_coords)
    hand_y_min = min(hand_y_coords)
    hand_y_max = max(hand_y_coords)
    
    hand_width = hand_x_max - hand_x_min
    hand_height = hand_y_max - hand_y_min
    hand_size = (hand_width + hand_height) / 2
    
    # 将大小转换为深度值 (0-1)
    hand_depth = min(1.0, hand_size * 2)
    
    return hand_depth, hand_size


def estimate_object_depth(box, frame_width, frame_height):
    """
    估计物体的深度
    基于物体框的大小：框越大 = 离摄像头越近
    返回 0-1，1 表示离摄像头最近
    """
    x1, y1, x2, y2 = box.xyxy[0]
    
    obj_width = (x2 - x1) / frame_width
    obj_height = (y2 - y1) / frame_height
    obj_size = (obj_width + obj_height) / 2
    
    # 将大小转换为深度值 (0-1)
    obj_depth = min(1.0, obj_size * 2)
    
    return obj_depth, obj_size


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    height, width, c = frame.shape
    
    # 使用实时画面
    annotated_frame = frame.copy()
    
    # 1. 鼠标检测
    results_yolo = yolo_model(frame)
    boxes = results_yolo[0].boxes
    
    mouse_detected = False
    mouse_x, mouse_y = None, None
    mouse_depth = None
    mouse_size = None
    
    for box in boxes:
        class_id = int(box.cls[0])
        class_name = yolo_model.names[class_id]
        
        if class_name.lower() == "mouse":
            mouse_detected = True
            x1, y1, x2, y2 = box.xyxy[0]
            mouse_x = (int(x1) + int(x2)) // 2
            mouse_y = (int(y1) + int(y2)) // 2
            
            # 估计鼠标深度
            mouse_depth, mouse_size = estimate_object_depth(box, width, height)
            
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
    hand_depth = None
    hand_size = None
    
    if detection_result.hand_landmarks:
        hand_detected = True
        for hand_landmarks in detection_result.hand_landmarks:
            middle_finger = hand_landmarks[12]
            hand_x = int(middle_finger.x * width)
            hand_y = int(middle_finger.y * height)
            
            # 估计手的深度
            hand_depth, hand_size = estimate_hand_depth(hand_landmarks, width, height)
            
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
    
    # 3. 深度信息显示
    start_y = 40
    
    # 标题
    cv2.putText(annotated_frame, "=== Depth Detection Test ===", 
               (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    start_y += 40
    
    # 手部信息
    if hand_detected and hand_depth is not None:
        cv2.putText(annotated_frame, f"Hand Size: {hand_size:.4f}", 
                   (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Hand Depth: {hand_depth:.4f}", 
                   (10, start_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        start_y += 75
    else:
        cv2.putText(annotated_frame, "Hand: Not detected", 
                   (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
        start_y += 40
    
    # 鼠标信息
    if mouse_detected and mouse_depth is not None:
        cv2.putText(annotated_frame, f"Mouse Size: {mouse_size:.4f}", 
                   (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"Mouse Depth: {mouse_depth:.4f}", 
                   (10, start_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        start_y += 75
    else:
        cv2.putText(annotated_frame, "Mouse: Not detected", 
                   (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
        start_y += 40
    
    # 深度对比结果
    cv2.putText(annotated_frame, "--- Result ---", 
               (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    start_y += 40
    
    if hand_detected and mouse_detected and hand_depth is not None and mouse_depth is not None:
        depth_diff = hand_depth - mouse_depth
        
        cv2.putText(annotated_frame, f"Depth Difference: {depth_diff:.4f}", 
                   (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 0), 2)
        start_y += 40
        
        # 判断是否在同一水平线
        depth_threshold = 0.05  # 深度差异阈值
        
        if abs(depth_diff) < depth_threshold:
            # 成功：在同一水平线
            cv2.putText(annotated_frame, "✓ SUCCESS!", 
                       (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(annotated_frame, "Same depth level!", 
                       (10, start_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 绘制成功指示框
            cv2.rectangle(annotated_frame, (5, start_y - 35), (320, start_y + 55), (0, 255, 0), 3)
        else:
            # 失败：不在同一水平线
            if depth_diff > 0:
                status = "Hand CLOSER"
                color = (0, 165, 255)  # 橙色
            else:
                status = "Hand FARTHER"
                color = (0, 0, 255)  # 红色
            
            cv2.putText(annotated_frame, "✗ DIFFERENT DEPTH", 
                       (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            cv2.putText(annotated_frame, status, 
                       (10, start_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        cv2.putText(annotated_frame, "Waiting for detection...", 
                   (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
    
    # 显示
    if frame_count % 3 == 0:
        cv2.imshow('Step 2: Depth Detection Test', annotated_frame)
    
    if cv2.waitKey(500) & 0xFF == ord('q'):
        print("Exited")
        break

cap.release()
cv2.destroyAllWindows()