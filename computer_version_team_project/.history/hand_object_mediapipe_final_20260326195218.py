import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

print("✓ MediaPipe 版本:", mp.__version__)

# 初始化 YOLO
yolo_model = YOLO('yolov8n.pt')

# 初始化 MediaPipe 手部检测
model_path = 'hand_landmarker.task'

if not os.path.exists(model_path):
    print(f"❌ 错误：找不到 {model_path} 文件")
    print("请先下载模型文件")
    exit()

print("⏳ 加载 MediaPipe 模型...")
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
print("✓ MediaPipe 模型加载完成\n")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

print("✓ 手部+物体检测已启动（MediaPipe）")
print(" 将你的手指向某个物体")
print("按 'q' 键退出...\n")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    height, width, c = frame.shape
    
    # 1. 物体检测
    results_yolo = yolo_model(frame)
    annotated_frame = results_yolo[0].plot()
    
    # 2. 手部检测（MediaPipe 新 API）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    
    # 获取手的位置
    hand_x, hand_y = None, None
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # 获取中指尖（第12个关键点）
            middle_finger = hand_landmarks[12]
            hand_x = int(middle_finger.x * width)
            hand_y = int(middle_finger.y * height)
            
            # 画出所有关键点（绿色点）
            for landmark in hand_landmarks:
                lx = int(landmark.x * width)
                ly = int(landmark.y * height)
                cv2.circle(annotated_frame, (lx, ly), 4, (0, 255, 0), -1)
            
            # 画出手指连接线（手的骨架）
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),           # 大拇指
                (0, 5), (5, 6), (6, 7), (7, 8),           # 食指
                (0, 9), (9, 10), (10, 11), (11, 12),      # 中指
                (0, 13), (13, 14), (14, 15), (15, 16),    # 无名指
                (0, 17), (17, 18), (18, 19), (19, 20),    # 小指
                (5, 9), (9, 13), (13, 17)                  # 连接各指
            ]
            
            for start, end in connections:
                start_point = (int(hand_landmarks[start].x * width), 
                              int(hand_landmarks[start].y * height))
                end_point = (int(hand_landmarks[end].x * width), 
                            int(hand_landmarks[end].y * height))
                cv2.line(annotated_frame, start_point, end_point, (255, 0, 0), 2)
            
            # 在中指尖画大红点
            cv2.circle(annotated_frame, (hand_x, hand_y), 12, (0, 0, 255), -1)
            cv2.putText(annotated_frame, "HAND", (hand_x - 50, hand_y - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 3. 分析手和物体关系
    boxes = results_yolo[0].boxes
    if hand_x is not None and len(boxes) > 0:
        box = boxes[0]
        
        x1, y1, x2, y2 = box.xyxy[0]
        obj_x = (int(x1) + int(x2)) // 2
        obj_y = (int(y1) + int(y2)) // 2
        
        class_id = int(box.cls[0])
        class_name = yolo_model.names[class_id]
        confidence = float(box.conf[0])
        
        # 计算距离
        dx = hand_x - obj_x
        dy = hand_y - obj_y
        distance = int(np.sqrt(dx**2 + dy**2))
        
        # 画连接线
        cv2.line(annotated_frame, (hand_x, hand_y), (obj_x, obj_y), (255, 255, 0), 3)
        
        # 显示物体信息
        cv2.putText(annotated_frame, f"Object: {class_name} ({confidence:.1%})", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(annotated_frame, f"Distance: {distance} px", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 判断方向
        threshold = 80
        direction = ""
        instruction = ""
        
        if abs(dx) > abs(dy):
            if dx < -threshold:
                direction = "Hand RIGHT of object"
                instruction = "← Move LEFT"
            elif dx > threshold:
                direction = "Hand LEFT of object"
                instruction = "→ Move RIGHT"
            else:
                direction = "Aligned horizontally"
        else:
            if dy < -threshold:
                direction = "Hand BELOW object"
                instruction = "↑ Move UP"
            elif dy > threshold:
                direction = "Hand ABOVE object"
                instruction = "↓ Move DOWN"
            else:
                direction = "Aligned vertically"
        
        cv2.putText(annotated_frame, direction, 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(annotated_frame, instruction, 
                   (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
        
        # 接近时提示
        if distance < 120:
            cv2.putText(annotated_frame, "✓ READY TO GRAB!", 
                       (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.rectangle(annotated_frame, (5, 185), (350, 220), (0, 255, 0), 3)
    else:
        if hand_x is None:
            cv2.putText(annotated_frame, "⚠ No hand detected", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if len(boxes) == 0:
            cv2.putText(annotated_frame, "⚠ No object detected", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 显示
    if frame_count % 3 == 0:
        cv2.imshow('Hand-Object Detection (MediaPipe)', annotated_frame)
    
    if cv2.waitKey(500) & 0xFF == ord('q'):
        print("已退出")
        break

cap.release()
cv2.destroyAllWindows()