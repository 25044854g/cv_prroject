import cv2
from ultralytics import YOLO

# 加载预训练的 YOLOv8 模型
# 第一次运行会自动下载模型（约100MB，需要点时间）
model = YOLO('yolov8n.pt')

# 打开摄像头
cap = cv2.VideoCapture(0)

print("✓ 物体检测已启动")
print("🎯 YOLO模型正在加载中...（第一次会慢一些）")
print("按 'q' 键退出...")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # 每隔5帧进行一次检测（提高速度）
    if frame_count % 5 == 0:
        # 进行物体检测
        results = model(frame)
        
        # 获取检测结果并绘制
        annotated_frame = results[0].plot()
        
        cv2.imshow('Object Detection', annotated_frame)
    else:
    
        cv2.imshow('Object Detection', frame)
    
    if cv2.waitKey(500) & 0xFF == ord('q'): 
        #keep the time is one , too short will cause the window to close immediately    
        # change the time to 1000ms, it will be more stable

        print("已退出")
        break

cap.release()
cv2.destroyAllWindows()