"""
Main application file
Imports and uses the HandObjectDetector, DepthDetector and VoiceDetector modules
"""

import cv2
import threading
import time
from hand_detection_module import HandObjectDetector
from depth_detection_module import DepthDetector
from voice_module import VoiceDetector

print("Loading models...")

# 先创建检测器获取 YOLO 类名列表
hand_detector = HandObjectDetector()
yolo_class_names = list(hand_detector.yolo_model.names.values())

# 语音输入目标物体
voice = VoiceDetector(yolo_class_names)
target_object = voice.get_target_object()

# 设置目标物体
hand_detector.target_object = target_object.lower()
depth_detector = DepthDetector(target_object=target_object)

print(f"Target: {target_object}")
print("Models loaded\n")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

print("✓ Hand-Object Detection with Depth Started")
print(f"Looking for: {target_object}")
print("Press 'q' to exit...\n")

frame_count = 0

# 语音导航：非阻塞播报 + 冷却时间
_speaking = False
_last_speak_time = 0
SPEAK_COOLDOWN = 3  # 每次语音间隔至少3秒

def speak_async(text):
    """非阻塞语音播报，避免卡住主循环"""
    global _speaking, _last_speak_time
    now = time.time()
    if _speaking or (now - _last_speak_time) < SPEAK_COOLDOWN:
        return
    _speaking = True
    _last_speak_time = now
    def _run():
        global _speaking
        try:
            voice.speak(text)
        finally:
            _speaking = False
    threading.Thread(target=_run, daemon=True).start()

last_guidance = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Process frame with hand detectorq
    hand_result = hand_detector.process_frame(frame)
    
    # Process frame with depth detector
    depth_result = depth_detector.process_frame(frame)
    
    # 获取两个检测器的结果
    annotated_frame = hand_result['annotated_frame']
    
    # ★★★ 简化判别逻辑：距离 < 100px 即视为已抓到 ★★★
    ready_to_grab = (
        hand_result['distance_2d'] is not None and 
        hand_result['distance_2d'] < 100                      # 主条件：距离足够近（100px以内）
    )
    
    # 显示综合结果
    if ready_to_grab:
        cv2.putText(annotated_frame, " GET IT! ", 
                   (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 5)
        cv2.rectangle(annotated_frame, (5, 200), (450, 245), (0, 255, 0), 4)
        if last_guidance != "grab":
            speak_async(f"Got the {target_object}!")
            last_guidance = "grab"
    else:
        # 构建语音导航指令
        guidance_parts = []
        
        if hand_result['distance_2d'] is None:
            # 手或物体未检测到
            if hand_result.get('direction') is None:
                guidance_parts.append(f"Looking for {target_object}")
        else:
            # 方向引导
            direction = hand_result.get('direction', '')
            if not hand_result['is_horizontally_aligned']:
                if 'RIGHT' in direction:
                    guidance_parts.append("move left")
                elif 'LEFT' in direction:
                    guidance_parts.append("move right")
            if not hand_result['is_vertically_aligned']:
                if 'ABOVE' in direction:
                    guidance_parts.append("move down")
                elif 'BELOW' in direction:
                    guidance_parts.append("move up")
            
            # 深度引导
            if not depth_result['is_same_depth']:
                if depth_result.get('depth_diff') is not None:
                    if depth_result['depth_diff'] > 0:
                        guidance_parts.append("move hand back")
                    else:
                        guidance_parts.append("move hand closer")
            
            # 距离引导
            if hand_result['distance_2d'] >= 120:
                guidance_parts.append("get closer")
        
        guidance = ", ".join(guidance_parts) if guidance_parts else "detecting"
        
        # 只在引导内容变化时语音播报
        if guidance != last_guidance and guidance != "detecting":
            speak_async(guidance)
            last_guidance = guidance
        
        # 显示未就位的原因
        reason = []
        if hand_result['distance_2d'] is None or hand_result['distance_2d'] >= 120:
            reason.append("distance")
        if not depth_result['is_same_depth']:
            reason.append("depth")
        if not hand_result['is_vertically_aligned']:
            reason.append("vertical")
        if not hand_result['is_horizontally_aligned']:
            reason.append("horizontal")
        
        reason_text = "not ready: " + "+".join(reason) if reason else "detecting"
        cv2.putText(annotated_frame, reason_text, 
                   (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Display
    if frame_count % 3 == 0:
        cv2.imshow('Hand-Object Detection with Depth', annotated_frame)
    
    if cv2.waitKey(500) & 0xFF == ord('q'):
        print("Exited")
        break

cap.release()
cv2.destroyAllWindows()