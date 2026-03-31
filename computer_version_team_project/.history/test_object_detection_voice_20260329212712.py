import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import speech_recognition as sr
import pyttsx3

# 语音合成通过 speak() 函数调用，每次重新初始化避免卡死

print("✓ MediaPipe version:", mp.__version__)

# 获取路径
base_dir = os.path.dirname(os.path.abspath(__file__))

# 初始化 YOLO
yolo_model = YOLO(os.path.join(base_dir, 'yolov8m.pt'))

# 初始化 MediaPipe 手部检测
model_path = os.path.join(base_dir, 'hand_landmarker.task')

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

# 常见别名映射到 YOLO COCO 类名
ALIAS_MAP = {
    "phone": "cell phone",
    "cellphone": "cell phone",
    "mobile": "cell phone",
    "mobile phone": "cell phone",
    "smartphone": "cell phone",
    "tv": "tv",
    "television": "tv",
    "monitor": "tv",
    "screen": "tv",
    "couch": "couch",
    "sofa": "couch",
    "bike": "bicycle",
    "motorbike": "motorcycle",
    "aeroplane": "airplane",
    "plane": "airplane",
    "remote control": "remote",
    "glasses": "wine glass",
    "glass": "wine glass",
    "mouse": "mouse",
    "computer mouse": "mouse",
    "mug": "cup",
}

# 获取 YOLO 所有可识别类名
yolo_class_names = [name.lower() for name in yolo_model.names.values()]

def extract_object_from_speech(speech_text):
    """从自然语言中提取 YOLO 可识别的物体名称"""
    words = speech_text.lower().strip()
    
    # 1. 先检查别名（优先匹配长别名）
    for alias in sorted(ALIAS_MAP.keys(), key=len, reverse=True):
        if alias in words:
            return ALIAS_MAP[alias]
    
    # 2. 检查 YOLO 类名（优先匹配多词类名如 "cell phone", "wine glass"）
    for class_name in sorted(yolo_class_names, key=len, reverse=True):
        if class_name in words:
            return class_name
    
    # 3. 逐词匹配
    for word in words.split():
        if word in yolo_class_names:
            return word
    
    # 4. 没匹配到，返回原文
    return words

def speak(text):
    """语音输出，每次重新初始化引擎避免卡死"""
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

def listen_once(prompt_text):
    """语音合成提示 + 监听一次语音输入，返回识别文本或 None"""
    print(prompt_text)
    speak(prompt_text)
    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=10)
        text = recognizer.recognize_google(audio).lower()
        print(f"You said: {text}")
        return text
    except (sr.WaitTimeoutError, sr.UnknownValueError):
        print("Could not hear clearly.")
        return None
    except sr.RequestError as e:
        print(f"Speech service error: {e}")
        return None

# 语音输入 + 确认循环
print("="*50)
print("Tell me what object you want to find")
print("Example: 'I want to find my cup'")
print("YOLO detectable: " + ", ".join(yolo_class_names[:15]) + " ...")
print("="*50)

target_object = None
MAX_RETRIES = 5

for attempt in range(MAX_RETRIES):
    # Step 1: 听用户说想找什么
    speech_text = listen_once("What object do you want to find? Please speak now.")
    if speech_text is None:
        continue
    
    # Step 2: 提取关键词
    extracted = extract_object_from_speech(speech_text)
    print(f"Extracted target: {extracted}")
    
    # Step 3: 语音复述原话 + 提取结果 + 询问确认
    confirm_prompt = f"I heard you say: {speech_text}. The target object is {extracted}. Do you want to find {extracted}? Please say yes or no."
    confirm_text = listen_once(confirm_prompt)
    if confirm_text is None:
        continue
    
    # Step 4: 判断回答
    if "yes" in confirm_text or "yeah" in confirm_text or "yep" in confirm_text or "correct" in confirm_text or "right" in confirm_text:
        target_object = extracted
        speak(f"OK, looking for {target_object}")
        print(f"Confirmed target: {target_object}\n")
        break
    else:
        print("Let's try again.\n")
        speak("OK, let's try again.")

if not target_object:
    print("Too many retries. Using default target: cell phone")
    target_object = "cell phone"

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