import cv2
import numpy as np
from ultralytics import YOLO

print("⏳ 加载模型中...")
# 初始化 YOLO 物体检测
yolo_detect = YOLO('yolov8n.pt')

# 初始化 YOLO 姿态检测（用于找手）
yolo_pose = YOLO('yolov8n-pose.pt')

print("✓ 模型加载完成\n")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

print("✓ 手部+物体检测已启动")
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
    results_detect = yolo_detect(frame)
    annotated_frame = results_detect[0].plot()
    
    # 2. 姿态检测（包含手的位置）
    results_pose = yolo_pose(frame)
    
    # 获取手的位置
    hand_x, hand_y = None, None
    if results_pose[0].keypoints is not None:
        keypoints = results_pose[0].keypoints[0].xy
        
        # YOLO pose 的关键点：
        # 10 = right wrist (右手腕)
        # 9 = left wrist (左手腕)
        
        # 优先用右手
        if len(keypoints) > 10:
            right_wrist = keypoints[10]
            if right_wrist[0] > 0:
                hand_x = int(right_wrist[0])
                hand_y = int(right_wrist[1])
        
        # 如果右手没有，用左手
        if hand_x is None and len(keypoints) > 9:
            left_wrist = keypoints[9]
            if left_wrist[0] > 0:
                hand_x = int(left_wrist[0])
                hand_y = int(left_wrist[1])
        
        # 绘制骨架
        annotated_frame = results_pose[0].plot()
    
    # 3. 分析手和物体关系
    boxes = results_detect[0].boxes
    if hand_x is not None and len(boxes) > 0:
        box = boxes[0]
        
        x1, y1, x2, y2 = box.xyxy[0]
        obj_x = (int(x1) + int(x2)) // 2
        obj_y = (int(y1) + int(y2)) // 2
        
        class_id = int(box.cls[0])
        class_name = yolo_detect.names[class_id]
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
        cv2.imshow('Hand-Object Detection (YOLO)', annotated_frame)
    
    if cv2.waitKey(500) & 0xFF == ord('q'):
        print("已退出")
        break

cap.release()
cv2.destroyAllWindows()