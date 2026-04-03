import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# 初始化模型
yolo_model = YOLO('yolov8n.pt')

# 初始化 MediaPipe 手部检测
try:
    # 尝试新版本API
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    print("使用 MediaPipe 新版本API")
except:
    # 如果失败，使用旧版本API
    print("使用 MediaPipe 旧版本API")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7
    )
    mp_drawing = mp.solutions.drawing_utils

# 打开摄像头
cap = cv2.VideoCapture(0)

print("✓ 手部+物体检测已启动（MediaPipe版本）")
print(" 将你的手指向某个物体")
print("按 'q' 键退出...")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    height, width, c = frame.shape
    
    # 1. 进行物体检测
    results_yolo = yolo_model(frame)
    annotated_frame = results_yolo[0].plot()
    
    # 2. 进行手部检测（使用MediaPipe）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hand = hands.process(rgb_frame)
    
    # 获取手的位置
    hand_x, hand_y = None, None
    if results_hand.multi_hand_landmarks:
        for hand_landmarks in results_hand.multi_hand_landmarks:
            # 获取中指尖的位置（第12个点）
            middle_finger = hand_landmarks.landmark[12]
            hand_x = int(middle_finger.x * width)
            hand_y = int(middle_finger.y * height)
            
            # 画出手部骨架
            mp_drawing.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 在中指尖画一个大红点
            cv2.circle(annotated_frame, (hand_x, hand_y), 15, (0, 0, 255), -1)
    
    # 3. 分析手和物体的空间关系
    boxes = results_yolo[0].boxes
    if hand_x is not None and len(boxes) > 0:
        # 获取第一个检测到的物体
        box = boxes[0]
        
        # 获取物体信息
        x1, y1, x2, y2 = box.xyxy[0]
        obj_x = (int(x1) + int(x2)) // 2
        obj_y = (int(y1) + int(y2)) // 2
        
        class_id = int(box.cls[0])
        class_name = yolo_model.names[class_id]
        confidence = float(box.conf[0])
        
        # 计算距离和方向
        dx = hand_x - obj_x
        dy = hand_y - obj_y
        distance = int(np.sqrt(dx**2 + dy**2))
        
        # 显示物体信息
        cv2.putText(annotated_frame, f"Object: {class_name} ({confidence:.1%})", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(annotated_frame, f"Distance: {distance} px", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 判断方向
        threshold = 60  # 方向判断的阈值
        
        direction = ""
        if abs(dx) > abs(dy):  # 主要是水平方向
            if dx < -threshold:
                direction = "-> Move LEFT"
                cv2.arrowedLine(annotated_frame, (hand_x, hand_y), (hand_x - 50, hand_y), 
                               (255, 0, 0), 3)
            elif dx > threshold:
                direction = "-> Move RIGHT"
                cv2.arrowedLine(annotated_frame, (hand_x, hand_y), (hand_x + 50, hand_y), 
                               (255, 0, 0), 3)
            else:
                direction = "Aligned horizontally"
        else:  # 主要是竖直方向
            if dy < -threshold:
                direction = "-> Move UP"
                cv2.arrowedLine(annotated_frame, (hand_x, hand_y), (hand_x, hand_y - 50), 
                               (255, 0, 0), 3)
            elif dy > threshold:
                direction = "-> Move DOWN"
                cv2.arrowedLine(annotated_frame, (hand_x, hand_y), (hand_x, hand_y + 50), 
                               (255, 0, 0), 3)
            else:
                direction = "Aligned vertically"
        
        cv2.putText(annotated_frame, direction, 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # 如果足够接近，提示可以抓取
        if distance < 100:
            cv2.putText(annotated_frame, "CLOSE! Ready to grab!", 
                       (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    else:
        if hand_x is None:
            cv2.putText(annotated_frame, "No hand detected", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if len(boxes) == 0:
            cv2.putText(annotated_frame, "No object detected", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 显示结果
    if frame_count % 3 == 0:
        cv2.imshow('Hand-Object Analysis', annotated_frame)
    
    if cv2.waitKey(500) & 0xFF == ord('q'):
        print("已退出")
        break

cap.release()
cv2.destroyAllWindows()