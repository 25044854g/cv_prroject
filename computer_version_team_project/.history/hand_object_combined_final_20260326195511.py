import cv2
import numpy as np
from ultralytics import YOLO

print("⏳ 加载模型中...")
yolo_detect = YOLO('yolov8n.pt')
yolo_pose = YOLO('yolov8n-pose.pt')
print("✓ 模型加载完成\n")

cap = cv2.VideoCapture(0)

if-4: 右肩、右肘、右腕、左肩、左肘
        # 5: 左腕 (index 9)
        # 更多细节：头、颈、肩、肘、腕等
        
        # 尝试获取右腕（通常在 index 10 或附近）
        if len(keypoints) > 10:
            right_wrist = keypoints[10]
            print(f"右腕坐标: {right_wrist}, 有效: {right_wrist[0] > 0 and right_wrist[1] > 0}")
            if right_wrist[0] > 10 and right_wrist[1] > 10:  # 验证坐标有效
                hand_x = int(right_wrist[0])
                hand_y = int(right_wrist[1])
        
        # 如果右手没有，用左手（通常在 index 9）
        if hand_x is None and len(keypoints) > 9:
            left_wrist = keypoints[9]
            print(f"左腕坐标: {left_wrist}, 有效: {left_wrist[0] > 0 and left_wrist[1] > 0}")
            if left_wrist[0] > 10 and left_wrist[1] > 10:
                hand_x = int(left_wrist[0])
                hand_y = int not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

print("✓ 手部+物体检测已启动")
print(" 伸出手指向物体")
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
    
    # 2. 姿态检测
    results_pose = yolo_pose(frame)
    
    # 获取手的位置（增强版）
    hand_x, hand_y = None, None
    if results_pose[0].keypoints is not None:
        keypoints = results_pose[0].keypoints[0].xy
        
        # 尝试找到任何可用的手关键点
        # 10 = right wrist, 9 = left wrist
        # 如果没有，用肩膀附近的点（11, 12）
        
        candidates = []
        
        # 右手(left_wrist[1])
        
        # 绘制骨架
        annotated_frame = results_pose[0].plot()
        
        # 调试：打印前5个关键点
        if frame_count % 30 == 0:
            print(f"\n前5个关键点:")
            for i in range(min(5, len(keypoints))):
                print(f"  {i}: {keypoints[i]}")
    
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
        # 在手的位置画红腕
        if len(keypoints) > 10 and keypoints[10][0] > 0:
            candidates.append(keypoints[10])
        
        # 左手腕
        if len(keypoints) > 9 and keypoints[9][0] > 0:
            candidates.append(keypoints[9])
        
        # 如果没有手腕，用手肘
        if not candidates:
            if len(keypoints) > 11 and keypoints[11][0] > 0:
                candidates.append(keypoints[11])
            if len(keypoints) > 12 and keypoints[12][0] > 0:
                candidates.append(keypoints[12])
        
        # 选择最靠下的点（通常是手）
        if candidates:
            hand_pos = max(candidates, key=lambda x: x[1])
            hand_x = int(hand_pos[0])
            hand_y = int(hand_pos[1])
        
        # 绘制骨架
        annotated_frame = results_pose[0].plot()
    
    # 3. 分析手和物体关系
    boxes = results_detect[0].boxes
    if hand_x is not None and len(boxes) > 0:
        box = boxes[0]
        
        x1, y1, x2, y2 = box.xyxy[0]
        obj_x = (int(x1) + int(x2)) // 2
        obj_y = (int(y1) + int点
        cv2.circle(annotated_frame, (hand_x, hand_y), 15, (0, 0, 255), -1)
        
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
        
        cv2.putText(annotated_frame, direction, (y2)) // 2
        
        class_id = int(box.cls[0])
        class_name = yolo_detect.names[class_id]
        confidence = float(box.conf[0])
        
        # 计算距离
        dx = hand_x - obj_x
        dy = hand_y - obj_y
        distance = int(np.sqrt(dx**2 + dy**2))
        
        # 画连接线
        cv2.line(annotated_frame, (hand_x, hand_y), (obj_x, obj_y), (255, 255, 0), 3)
        cv2.circle(annotated_frame, (hand_x, hand_y), 10, (0, 0, 255), -1)
        
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
                instruction = "←
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