import cv2
import numpy as np
from ultralytics import YOLO

# 初始化模型
yolo_model = YOLO('yolov8n.pt')

# 打开摄像头
cap = cv2.VideoCapture(0)

print("✓ 手部+物体检测已启动")
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
    results = yolo_model(frame)
    
    # 2. 进行手部检测（用皮肤颜色）
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 获取手的位置
    hand_x, hand_y = None, None
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:  # 过滤太小的区域
            x, y, w, h = cv2.boundingRect(largest_contour)
            hand_x = x + w // 2  # 手的中心x
            hand_y = y + h // 2  # 手的中心y
            
            # 画出手的位置（红色圆点）
            cv2.circle(frame, (hand_x, hand_y), 10, (0, 0, 255), -1)
            cv2.putText(frame, "Hand", (hand_x - 30, hand_y - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 3. 绘制物体检测结果
    annotated_frame = results[0].plot()
    
    # 4. 分析手和物体的空间关系
    boxes = results[0].boxes
    if hand_x is not None and len(boxes) > 0:
        for idx, box in enumerate(boxes):
            # 获取物体信息
            x1, y1, x2, y2 = box.xyxy[0]
            obj_x = (int(x1) + int(x2)) // 2  # 物体中心x
            obj_y = (int(y1) + int(y2)) // 2  # 物体中心y
            
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]
            
            # 判断相对位置
            dx = hand_x - obj_x
            dy = hand_y - obj_y
            distance = int(np.sqrt(dx**2 + dy**2))
            
            # 显示关系信息
            info = f"Object: {class_name}"
            cv2.putText(annotated_frame, info, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            info2 = f"Distance: {distance} px"
            cv2.putText(annotated_frame, info2, (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 判断方向
            if abs(dx) > abs(dy):  # 水平方向差异更大
                if dx < -50:
                    direction = "Hand RIGHT of object -> Move LEFT"
                elif dx > 50:
                    direction = "Hand LEFT of object -> Move RIGHT"
                else:
                    direction = "Hand aligned horizontally"
            else:  # 竖直方向差异更大
                if dy < -50:
                    direction = "Hand BELOW object -> Move UP"
                elif dy > 50:
                    direction = "Hand ABOVE object -> Move DOWN"
                else:
                    direction = "Hand aligned vertically"
            
            cv2.putText(annotated_frame, direction, (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # 如果足够接近
            if distance < 100:
                cv2.putText(annotated_frame, "CLOSE! Can grab!", (10, 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    # 显示结果
    if frame_count % 5 == 0:
        cv2.imshow('Hand-Object Analysis', annotated_frame)
    
    if cv2.waitKey(500) & 0xFF == ord('q'):
        print("已退出")
        break

cap.release()
cv2.destroyAllWindows()