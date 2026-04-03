import cv2
import math

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
else:
    print("摄像头已打开")
    print("矩形框会随着时间动起来")
    print("按 'q' 键退出...")
    
    frame_count = 0  # 计数器，用来追踪时间
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width = frame.shape[:2]
        frame_count += 1
        
        # 让坐标随着时间改变（利用sin和cos函数制造波形运动）
        offset_x = int(100 * math.sin(frame_count * 0.05))  # 水平摇晃
        offset_y = int(50 * math.cos(frame_count * 0.05))   # 竖直摇晃
        
        x1 = 200 + offset_x
        y1 = 200 + offset_y
        x2 = 400 + offset_x
        y2 = 400 + offset_y
        
        # 画动起来的矩形框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 显示帧数（可选）
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Moving Rectangle', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("已退出")
            break
    
    cap.release()
    cv2.destroyAllWindows()