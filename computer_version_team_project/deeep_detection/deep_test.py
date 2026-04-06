import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

print("MediaPipe version:", mp.__version__)

# Resolve paths relative to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

# Initialize YOLO with Medium model for better accuracy
yolo_model = YOLO(os.path.join(project_dir, 'yolov8m.pt'))

# Initialize MediaPipe hand detection
model_path = os.path.join(project_dir, 'hand_landmarker.task')

if not os.path.exists(model_path):
    print(f"ERROR: Cannot find {model_path} file")
    print("Please download the model file first")
    exit()

print("Loading MediaPipe model...")
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
print("MediaPipe model loaded\n")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

print("Hand-Object Detection with Depth Started (MediaPipe + YOLOv8m)")
print("Point your hand toward an object (NOT a person)")
print("Press 'q' to exit...\n")

frame_count = 0
PERSON_CLASS_ID = 0
DEPTH_THRESHOLD = 0.1  # 深度差异阈值


def get_hand_depth(hand_landmarks, frame_width, frame_height):
    """
    估计手的深度，基于手的大小
    手越大 = 离摄像头越近（深度值越高）
    """
    if not hand_landmarks:
        return None
    
    # 获取手的所有坐标
    hand_x_coords = [lm.x for lm in hand_landmarks]
    hand_y_coords = [lm.y for lm in hand_landmarks]
    
    # 计算手的边界框
    hand_x_min = min(hand_x_coords)
    hand_x_max = max(hand_x_coords)
    hand_y_min = min(hand_y_coords)
    hand_y_max = max(hand_y_coords)
    
    # 计算手的大小（归一化）
    hand_width = hand_x_max - hand_x_min
    hand_height = hand_y_max - hand_y_min
    hand_size = (hand_width + hand_height) / 2
    
    # 手的深度估计：0-1，1表示最近
    hand_depth = min(1.0, hand_size * 2)
    
    return {
        'depth': hand_depth,
        'size': hand_size,
        'x_min': hand_x_min,
        'x_max': hand_x_max,
        'y_min': hand_y_min,
        'y_max': hand_y_max
    }


def get_object_depth(box, frame_width, frame_height):
    """
    估计物体的深度，基于物体的大小
    物体框越大 = 离摄像头越近（深度值越高）
    """
    x1, y1, x2, y2 = box.xyxy[0]
    
    # 计算物体的相对大小
    obj_width = (x2 - x1) / frame_width
    obj_height = (y2 - y1) / frame_height
    obj_size = (obj_width + obj_height) / 2
    
    # 物体深度估计：0-1，1表示最近
    obj_depth = min(1.0, obj_size * 2)
    
    return {
        'depth': obj_depth,
        'size': obj_size,
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2
    }


def is_hand_in_front_of_object(hand_depth_info, object_depth_info):
    """
    判断手是否在物体前面
    手的深度 > 物体深度 + 阈值 = 手在前面
    """
    hand_depth = hand_depth_info['depth']
    object_depth = object_depth_info['depth']
    
    # 深度差异（正数 = 手离摄像头更近）
    depth_diff = hand_depth - object_depth
    
    # 手必须明显在前面
    is_in_front = depth_diff > DEPTH_THRESHOLD
    
    return {
        'depth_diff': depth_diff,
        'is_in_front': is_in_front,
        'hand_depth': hand_depth,
        'object_depth': object_depth
    }


def is_hand_over_object(hand_x, hand_y, hand_depth_info, obj_depth_info):
    """
    判断手是否在物体上方（2D位置）
    """
    x1, y1, x2, y2 = obj_depth_info['x1'], obj_depth_info['y1'], obj_depth_info['x2'], obj_depth_info['y2']
    
    obj_width = x2 - x1
    obj_height = y2 - y1
    
    # 增加20%的检测范围
    margin = 0.2
    left = int(x1 - obj_width * margin)
    right = int(x2 + obj_width * margin)
    top = int(y1 - obj_height * margin)
    bottom = int(y2 + obj_height * margin)
    
    # 检查手是否在物体范围内
    is_over = (left <= hand_x <= right) and (top <= hand_y <= bottom)
    
    return is_over


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    height, width, c = frame.shape
    
    # 1. 物体检测
    results_yolo = yolo_model(frame)
    annotated_frame = results_yolo[0].plot()
    
    # 2. 手部检测
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    
    # 获取手的位置和深度
    hand_x, hand_y = None, None
    hand_depth_info = None
    
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # 获取中指尖（landmark 12）
            middle_finger = hand_landmarks[12]
            hand_x = int(middle_finger.x * width)
            hand_y = int(middle_finger.y * height)
            
            # 获取手的深度估计
            hand_depth_info = get_hand_depth(hand_landmarks, width, height)
            
            # 绘制所有关键点（绿色点）
            for landmark in hand_landmarks:
                lx = int(landmark.x * width)
                ly = int(landmark.y * height)
                cv2.circle(annotated_frame, (lx, ly), 4, (0, 255, 0), -1)
            
            # 绘制手部骨架
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17)
            ]
            
            for start, end in connections:
                start_point = (int(hand_landmarks[start].x * width), 
                              int(hand_landmarks[start].y * height))
                end_point = (int(hand_landmarks[end].x * width), 
                            int(hand_landmarks[end].y * height))
                cv2.line(annotated_frame, start_point, end_point, (255, 0, 0), 2)
            
            # 在中指尖绘制红色大圆
            cv2.circle(annotated_frame, (hand_x, hand_y), 12, (0, 0, 255), -1)
            cv2.putText(annotated_frame, "HAND", (hand_x - 50, hand_y - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 3. 分析手和物体的关系（带深度检测）
    boxes = results_yolo[0].boxes
    target_box = None
    
    # 过滤掉person，只检测其他物体
    if hand_x is not None and len(boxes) > 0:
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]
            
            # 跳过person
            if class_id == PERSON_CLASS_ID or class_name.lower() == "person":
                continue
            
            target_box = box
            break
    
    # 处理目标物体（如果找到且不是人）
    if hand_x is not None and target_box is not None and hand_depth_info is not None:
        box = target_box
        
        x1, y1, x2, y2 = box.xyxy[0]
        obj_x = (int(x1) + int(x2)) // 2
        obj_y = (int(y1) + int(y2)) // 2
        
        class_id = int(box.cls[0])
        class_name = yolo_model.names[class_id]
        confidence = float(box.conf[0])
        
        # 获取物体的深度估计
        object_depth_info = get_object_depth(box, width, height)
        
        # 估计深度关系
        depth_result = is_hand_in_front_of_object(hand_depth_info, object_depth_info)
        
        # 检查手是否在物体上方（2D位置）
        is_over = is_hand_over_object(hand_x, hand_y, hand_depth_info, object_depth_info)
        
        # 绘制目标框（紫红色）
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                     (255, 0, 255), 3)
        
        # 计算2D距离
        dx = hand_x - obj_x
        dy = hand_y - obj_y
        distance_2d = int(np.sqrt(dx**2 + dy**2))
        
        # 在手和物体之间画线
        cv2.line(annotated_frame, (hand_x, hand_y), (obj_x, obj_y), (255, 255, 0), 3)
        
        # 显示物体信息
        cv2.putText(annotated_frame, f"Object: {class_name} ({confidence:.1%})", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(annotated_frame, f"Distance 2D: {distance_2d} px", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 显示深度信息
        cv2.putText(annotated_frame, 
                   f"Hand Depth: {hand_depth_info['depth']:.2f}", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 1)
        
        cv2.putText(annotated_frame, 
                   f"Object Depth: {object_depth_info['depth']:.2f}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 1)
        
        cv2.putText(annotated_frame, 
                   f"Depth Diff: {depth_result['depth_diff']:.3f}", 
                   (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 1)
        
        # 判断方向
        threshold = 80
        direction = ""
        instruction = ""
        
        if abs(dx) > abs(dy):
            if dx < -threshold:
                direction = "Hand is RIGHT of object"
                instruction = "Move LEFT to object"
                color = (0, 165, 255)
            elif dx > threshold:
                direction = "Hand is LEFT of object"
                instruction = "Move RIGHT to object"
                color = (0, 165, 255)
            else:
                direction = "Aligned horizontally"
                instruction = ""
                color = (0, 255, 0)
        else:
            if dy > -threshold:
                direction = "Hand is BELOW object"
                instruction = "Move UP to object"
                color = (0, 165, 255)
            elif dy < threshold:
                direction = "Hand is ABOVE object"
                instruction = "Move DOWN to object"
                color = (0, 165, 255)
            else:
                direction = "Aligned vertically"
                instruction = ""
                color = (0, 255, 0)
        
        cv2.putText(annotated_frame, direction, 
                   (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(annotated_frame, instruction, 
                   (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        ready_to_grab = (
            distance_2d < 120 and              # 1. 2D距离小于120像素
            depth_result['is_in_front'] and   # 2. 手在物体前面（深度）
            is_over                           # 3. 手在物体上方（位置）
        )
        
        if ready_to_grab:
            cv2.putText(annotated_frame, "✓ READY TO GRAB!", 
                       (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.rectangle(annotated_frame, (5, 205), (380, 240), (0, 255, 0), 3)
        else:
            # 显示原因
            reason = ""
            if distance_2d >= 120:
                reason = "Too far"
            elif not depth_result['is_in_front']:
                reason = "Hand BEHIND object"
            elif not is_over:
                reason = "Not aligned"
            
            if reason:
                cv2.putText(annotated_frame, f"✗ NOT READY: {reason}", 
                           (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 调试输出
        if frame_count % 30 == 0:
            print(f"\n[Frame {frame_count}]")
            print(f"  Hand size: {hand_depth_info['size']:.3f} (depth: {hand_depth_info['depth']:.2f})")
            print(f"  Object size: {object_depth_info['size']:.3f} (depth: {object_depth_info['depth']:.2f})")
            print(f"  Depth diff: {depth_result['depth_diff']:.3f} | In front: {depth_result['is_in_front']}")
            print(f"  Position over: {is_over} | Distance 2D: {distance_2d}")
            print(f"  Ready: {ready_to_grab}")
    else:
        if hand_x is None:
            cv2.putText(annotated_frame, "No hand detected", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 显示
    if frame_count % 3 == 0:
        cv2.imshow('Hand-Object Detection with Depth', annotated_frame)
    
    if cv2.waitKey(500) & 0xFF == ord('q'):
        print("Exited")
        break

cap.release()
cv2.destroyAllWindows()