import cv2
import numpy as np
from ultralytics import YOLO

# 初始化 YOLO（既用于物体检测，也用于手部检测）
yolo_model = YOLO('yolov8n.pt')
yolo_pose = YOLO('yolov8n-pose.pt')  # 用于姿态识别，可以检测手

# 打开摄像头
cap = cv2.VideoCapture(0)

print("✓ 手部+物体检测已启动（YOLO Pose 版本）")
print("👋 将你的手指向某个物体")
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
    
    # 2. 手部和姿态检测
    results_pose = yolo_pose(frame)
    
    # 获取手的位置（从姿态检测中获取）
    hand_x, hand_y = None, None
    if results_pose[0].keypoints is not None:
        keypoints = results_pose[0].keypoints[0].xy
        
        # YOLO 姿态模型的关键点包括手腕、肘部等
        # 我们用右手腕（通常是第10个点）
        if len(keypoints) > 10:
            right_wrist = keypoints[10]
            if right_wrist[0] > 0:  # 有效的关键点
                hand_x = int(right_wrist[0])
                hand_y = int(right_wrist[1])
        
        # 绘制姿态骨架
        annotated_frame = results_pose[0].plot()
    
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
        
        # 显示信息
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
                direction = "Horizontally aligned"
        else:
            if dy < -threshold:
                direction = "Hand BELOW object"
                instruction = "↑ Move UP"
            elif dy > threshold:
                direction = "Hand ABOVE object"
                instruction = "↓ Move DOWN"
            else:
                direction = "Vertically aligned"
        
        cv2.putText(annotated_frame, direction, 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(annotated_frame, instruction, 
                   (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
        
        if distance < 120:
            cv2.putText(annotated_frame, "✓ READY TO GRAB!", 
                       (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.rectangle(annotated_frame, (5, 185), (350, 220), (0, 255, 0), 3)
    
    # 显示
    if frame_count % 3 == 0:
        cv2.imshow('Hand-Object Detection', annotated_frame)
    
    if cv2.waitKey(500) & 0xFF == ord('q'):
        print("已退出")
        break

cap.release()
cv2.destroyAllWindows()