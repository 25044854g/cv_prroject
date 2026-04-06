import cv2
from ultralytics import YOLO
# have problem with the yolov8n.pt model, so I change it to yolov8m.pt, it will be more accurate but slower
model = YOLO('yolov8m.pt')
cap = cv2.VideoCapture(0)

print("物体检测已启动（详细信息版）")
print("按 'q' 键退出...")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # 每隔5帧进行检测
    if frame_count % 5 == 0:
        results = model(frame)
        annotated_frame = results[0].plot()
        
        # 提取检测信息
        boxes = results[0].boxes
        
        if len(boxes) > 0:
            print(f"\n--- 第 {frame_count} 帧 ---")
            for idx, box in enumerate(boxes):
                # 获取类别ID和名称
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # 获取置信度
                confidence = float(box.conf[0])
                
                # 获取坐标
                x1, y1, x2, y2 = box.xyxy[0]
                
                print(f"物体 {idx+1}: {class_name}")
                print(f"  置信度: {confidence:.2%}")
                print(f"  坐标: ({int(x1)}, {int(y1)}) 到 ({int(x2)}, {int(y2)})")
        
        cv2.imshow('Object Detection', annotated_frame)
    else:
        cv2.imshow('Object Detection', frame)
    
    if cv2.waitKey(400) & 0xFF == ord('q'):
        print("已退出")
        break

cap.release()
cv2.destroyAllWindows()