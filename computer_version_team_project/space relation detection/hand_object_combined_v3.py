import cv2
import numpy as np
from ultralytics import YOLO

# 初始化 YOLO 模型
yolo_model = YOLO('yolov8n.pt')

# 打开摄像头
cap = cv2.VideoCapture(0)

print("✓ 手部+物体检测已启动（OpenCV版本）")
print("👋 将你的手指向某个物体")
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
    
    # 2. 进行手部检测（用皮肤颜色 + 改进的参数）
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 更宽泛的皮肤颜色范围（适应不同光线）
    lower_skin1 = np.array([0, 10, 60], dtype=np.uint8)
    upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
    
    lower_skin2 = np.array([170, 10, 60], dtype=np.uint8)
    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
    
    mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # 进行形态学操作来清理噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # 找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 获取手的位置
    hand_x, hand_y = None, None
    if contours:
        # 获取最大的轮廓（应该是手）
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 800:  # 过滤太小的区域
            # 获取轮廓的矩形边界
            x, y, w, h = cv2.boundingRect(largest_contour)
            hand_x = x + w // 2  # 手的中心x
            hand_y = y + h // 2  # 手的中心y
            
            # 在原图上画出手的位置
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.circle(annotated_frame, (hand_x, hand_y), 15, (0, 0, 255), -1)
            cv2.putText(annotated_frame, "HAND", (hand_x - 40, hand_y - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
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
        
        # 在画面上画一条连接手和物体的线
        cv2.line(annotated_frame, (hand_x, hand_y), (obj_x, obj_y), (255, 255, 0), 2)
        
        # 显示物体信息
        cv2.putText(annotated_frame, f"Object: {class_name} ({confidence:.1%})", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(annotated_frame, f"Distance: {distance} px", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 判断方向
        threshold = 80  # 方向判断的阈值
        
        direction = ""
        instruction = ""
        
        if abs(dx) > abs(dy):  # 主要是水平方向
            if dx < -threshold:
                direction = "Hand is RIGHT of object"
                instruction = "-> Move LEFT"
            elif dx > threshold:
                direction = "Hand is LEFT of object"
                instruction = "-> Move RIGHT"
            else:
                direction = "Aligned horizontally"
                instruction = "Move vertically"
        else:  # 主要是竖直方向
            if dy < -threshold:
                direction = "Hand is BELOW object"
                instruction = "-> Move UP"
            elif dy > threshold:
                direction = "Hand is ABOVE object"
                instruction = "-> Move DOWN"
            else:
                direction = "Aligned vertically"
                instruction = "Move horizontally"
        
        cv2.putText(annotated_frame, direction, 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        cv2.putText(annotated_frame, instruction, 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
        
        # 如果足够接近，提示可以抓取
        if distance < 120:
            cv2.putText(annotated_frame, "CLOSE! Ready to grab!", 
                       (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.rectangle(annotated_frame, (5, 175), (400, 210), (0, 255, 0), 3)
    else:
        if hand_x is None:
            cv2.putText(annotated_frame, "No hand detected - check lighting", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if len(boxes) == 0:
            cv2.putText(annotated_frame, "No object detected", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 显示结果（每3帧显示一次，避免闪烁）
    if frame_count % 3 == 0:
        cv2.imshow('Hand-Object Spatial Analysis', annotated_frame)
    
    if cv2.waitKey(500) & 0xFF == ord('q'):
        print("已退出")
        break

cap.release()
cv2.destroyAllWindows()