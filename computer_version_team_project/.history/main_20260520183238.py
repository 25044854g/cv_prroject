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


SUPPORTED_TARGET_OBJECTS = [
    "cell phone",
    "mouse",
    "cup",
    "bottle",
    "book",
    "remote",
    "keyboard",
    "laptop",
]

print("Loading models...")

hand_detector = HandObjectDetector()
available_class_names = {name.lower() for name in hand_detector.yolo_model.names.values()}
yolo_class_names = [
    name for name in SUPPORTED_TARGET_OBJECTS if name.lower() in available_class_names
]

if not yolo_class_names:
    raise RuntimeError("No supported target objects were found in the YOLO model class list.")

voice = VoiceDetector(yolo_class_names)
target_object = voice.get_target_object()

hand_detector.target_object = target_object.lower()
depth_detector = DepthDetector(target_object=target_object)

print(f"Target: {target_object}")
print("Models loaded\n")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

print("Hand-Object Detection with Depth Started")
print(f"Looking for: {target_object}")
print("Press 'q' to exit...\n")

frame_count = 0

_speaking = False
_last_speak_time = 0
SPEAK_COOLDOWN = 1
_last_grab_speak_time = 0
GRAB_REPEAT_SECONDS = 3


def speak_async(text):
    """Non-blocking voice output with cooldown to prevent freezing main loop"""
    global _speaking, _last_speak_time
    now = time.time()
    if _speaking or (now - _last_speak_time) < SPEAK_COOLDOWN:
        return False
    _speaking = True
    _last_speak_time = now
    
    def _run():
        global _speaking
        try:
            voice.speak(text)
        finally:
            _speaking = False
    
    threading.Thread(target=_run, daemon=True).start()
    return True


last_guidance = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    hand_result = hand_detector.process_frame(frame)
    depth_result = depth_detector.process_frame(frame)
    annotated_frame = hand_result['annotated_frame']
    
    if hand_result['distance_2d'] is not None and hand_result['distance_2d'] < 100:
        conditions_met = sum([
            hand_result['distance_2d'] < 100,
            depth_result.get('is_same_depth', False),
            hand_result['is_vertically_aligned'],
            hand_result['is_horizontally_aligned']
        ])
        ready_to_grab = conditions_met >= 2
    else:
        ready_to_grab = False
    
    if ready_to_grab:
        cv2.putText(annotated_frame, " GET IT! ", 
                   (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 5)
        cv2.rectangle(annotated_frame, (5, 200), (450, 245), (0, 255, 0), 4)

        grab_prompt = f"Got the {target_object}! Now lift it up!"
        current_time = time.time()

        if last_guidance != "grab":
            if speak_async(grab_prompt):
                _last_grab_speak_time = current_time
            last_guidance = "grab"
        elif current_time - _last_grab_speak_time >= GRAB_REPEAT_SECONDS:
            if speak_async(grab_prompt):
                _last_grab_speak_time = current_time
    else:
        guidance_parts = []
        
        if hand_result['distance_2d'] is None:
            if hand_result.get('direction') is None:
                guidance_parts.append(f"Looking for {target_object}")
        else:
            dist = hand_result['distance_2d']
            dist_remaining = 100 - dist
            dist_color = (0, 255, 0) if dist < 100 else (0, 165, 255)
            
            cv2.putText(annotated_frame, f"Distance: {dist}px / Target: 100px", 
                       (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.0, dist_color, 2)
            
            if dist < 100:
                cv2.putText(annotated_frame, f"Remaining: {dist_remaining}px to grab!", 
                           (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(annotated_frame, f"Bring hand closer: {dist_remaining}px more needed", 
                           (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            
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
            
            if not depth_result['is_same_depth']:
                if depth_result.get('depth_diff') is not None:
                    if depth_result['depth_diff'] > 0:
                        guidance_parts.append("move hand back")
                    else:
                        guidance_parts.append("move hand closer")
            
            if hand_result['distance_2d'] >= 120:
                guidance_parts.append("get closer")
        
        guidance = ", ".join(guidance_parts) if guidance_parts else "detecting"
        
        if guidance != "detecting" and frame_count % 9 == 0:
            speak_async(guidance)
        
        last_guidance = guidance
        
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
    
    if frame_count % 3 == 0:
        cv2.imshow('Hand-Object Detection with Depth', annotated_frame)
    
    if cv2.waitKey(500) & 0xFF == ord('q'):
        print("Exited")
        break

cap.release()
cv2.destroyAllWindows()
