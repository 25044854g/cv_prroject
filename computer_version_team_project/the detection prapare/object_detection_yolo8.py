import cv2
from ultralytics import YOLO

# Load pre-trained YOLOv8 model
# First run will automatically download the model (about 100MB, takes some time)
model = YOLO('yolov8n.pt')

# Open camera
cap = cv2.VideoCapture(0)

print("物体检测已启动")
print(" YOLO模型正在加载中...（第一次会慢一些）")
print("按 'q' 键退出...")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Detect every 5 frames (improve speed)
    if frame_count % 5 == 0:
        # Perform object detection
        results = model(frame)
        
        # Get detection results and draw
        annotated_frame = results[0].plot()
        
        cv2.imshow('Object Detection', annotated_frame)
    else:
    
        cv2.imshow('Object Detection', frame)
    
    if cv2.waitKey(500) & 0xFF == ord('q'): 
        #keep the time is one , too short will cause the window to close immediately    
        # change the time to 500 ms, it will be more stable

        print("已退出")
        break

cap.release()
cv2.destroyAllWindows()