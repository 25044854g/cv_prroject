import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import speech_recognition as sr

print("✓ MediaPipe version:", mp.__version__)

# 获取路径
base_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(base_dir)

# 初始化 YOLO
yolo_model = YOLO(os.path.join(project_dir, 'yolov8m.pt'))

# 初始化 MediaPipe 手部检测
model_path = os.path.join(project_dir, 'hand_landmarker.task')

if not os.path.exists(model_path):
    print(f"ERROR: Cannot find {model_path} file")
    exit()

print("Loading MediaPipe model...")
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
print("✓ MediaPipe model loaded\n")

# 初始化语音识别
print("Initializing speech recognition...")
recognizer = sr.Recognizer()
microphone = sr.Microphone()

print("✓ Speech recognition ready\n")

# 获取语音输入的物体名称
print("="*50)
print("Please speak the object name you want to find")
print("Examples: cup, bottle, cola, water, glass, phone, laptop, book, pen, apple")
print("="*50)

target_object = None
try:
    print("Listening... Please speak now...")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, timeout=10)
    
    print("Processing speech...")
    target_object = recognizer.recognize_google(audio).lower()
    print(f"✓ You said: {target_object}\n")
    
except sr.UnknownValueError:
    print("ERROR: Could not understand speech. Please try again.")
    exit()
except sr.RequestError as e:
    print(f"ERROR: Speech recognition service error: {e}")
    exit()

if not target_object:
    print("ERROR: Please say an object name")
    exit()

print(f"✓ Looking for: {target_object}\n")
print("Press 'q' to exit...\n")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

frame_count = 0
found_objects = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    height, width, c = frame.shape
    
    # 使用实时画面
    annotated_frame = frame.copy()
    
    # 物体检测
    results_yolo = yolo_model(frame)
    boxes = results_yolo[0].boxes
    
    # 显示目标物体
    cv2.putText(annotated_frame, f"Target: {target_object.upper()}", 
               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
    
    target_found = False
    detected_objects_list = []
    
    # 遍历所有检测到的物体
    for box in boxes:
        class_id = int(box.cls[0])
        class_name = yolo_model.names[class_id]
        confidence = float(box.conf[0])
        
        x1, y1, x2, y2 = box.xyxy[0]
        
        # 记录检测到的物体
        detected_objects_list.append({
            'name': class_name,
            'confidence': confidence,
            'box': (int(x1), int(y1), int(x2), int(y2))
        })
        
        # 检查是否是目标物体
        if class_name.lower() == target_object:
            target_found = True
            found_objects.add(class_name.lower())
            
            # 绘制目标物体框（绿色）
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                         (0, 255, 0), 3)
            cv2.putText(annotated_frame, f"{class_name} ({confidence:.1%})", 
                       (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            # 绘制其他物体框（蓝色）
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                         (255, 0, 0), 2)
            cv2.putText(annotated_frame, f"{class_name} ({confidence:.1%})", 
                       (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
    
    # 显示检测状态
    start_y = 100
    
    if target_found:
        cv2.putText(annotated_frame, f"✓ Found: {target_object.upper()}", 
                   (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        start_y += 50
    else:
        cv2.putText(annotated_frame, f"✗ Not found: {target_object.upper()}", 
                   (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        start_y += 50
    
    # 显示所有检测到的物体
    cv2.putText(annotated_frame, "Detected objects:", 
               (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    start_y += 35
    
    if detected_objects_list:
        for i, obj in enumerate(detected_objects_list[:5]):  # 只显示前5个
            obj_text = f"{i+1}. {obj['name']} ({obj['confidence']:.1%})"
            if obj['name'].lower() == target_object:
                color = (0, 255, 0)
            else:
                color = (200, 200, 200)
            cv2.putText(annotated_frame, obj_text, 
                       (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
            start_y += 30
    else:
        cv2.putText(annotated_frame, "No objects detected", 
                   (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1)
    
    # 显示
    if frame_count % 3 == 0:
        cv2.imshow(f'Object Detection - Voice Input: {target_object.upper()}', annotated_frame)
    
    if cv2.waitKey(500) & 0xFF == ord('q'):
        print("\nExited")
        break

cap.release()
cv2.destroyAllWindows()

# 显示统计
print("\n" + "="*50)
print("Detection Summary:")
print(f"Target object: {target_object}")
print(f"Found: {target_object in found_objects}")
if found_objects:
    print(f"All objects found: {', '.join(found_objects)}")
else:
    print("No target objects found during session")
print("="*50)