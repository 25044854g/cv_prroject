"""
Main application file
Imports and uses the HandObjectDetector and DepthDetector modules
"""

import cv2
from hand_detection_module import HandObjectDetector
from depth_detection_module import DepthDetector

print("✓ MediaPipe version:", __import__('mediapipe').__version__)
print("Loading MediaPipe model...")

# Initialize detectors
hand_detector = HandObjectDetector()
depth_detector = DepthDetector()

print("✓ MediaPipe model loaded\n")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

print("✓ Hand-Object Detection with Depth Started (MediaPipe + YOLOv8m)")
print("Point your hand toward an object (NOT a person)")
print("Press 'q' to exit...\n")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Process frame with hand detector
    hand_result = hand_detector.process_frame(frame)
    
    # Process frame with depth detector
    depth_result = depth_detector.process_frame(frame)
    
    # 获取两个检测器的结果
    annotated_frame = hand_result['annotated_frame']
    
    # ★★★ 综合判别逻辑在这里 ★★★
    # 四个条件都必须满足：
    # 1. 距离 < 120px
    # 2. 深度相同
    # 3. 上下对齐
    # 4. 左右对齐
    
    ready_to_grab = (
        hand_result['distance_2d'] is not None and 
        hand_result['distance_2d'] < 120 and                  # 条件1：距离足够近
        depth_result['is_same_depth'] and                    # 条件2：深度相同
        hand_result['is_vertically_aligned'] and             # 条件3：上下对齐
        hand_result['is_horizontally_aligned']               # 条件4：左右对齐
    )
    
    # 显示综合结果
    if ready_to_grab:
        cv2.putText(annotated_frame, " READY TO GRAB! ", 
                   (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        cv2.rectangle(annotated_frame, (5, 200), (500, 245), (0, 255, 0), 3)
    else:
        # 显示未就位的原因
        reason = []
        if hand_result['distance_2d'] is None or hand_result['distance_2d'] >= 120:
            reason.append("desteance")
        if not depth_result['is_same_depth']:
            reason.append("depth")
        if not hand_result['is_vertically_aligned']:
            reason.append("vertical")
        if not hand_result['is_horizontally_aligned']:
            reason.append("horizontal")
        
        reason_text = "not ture: " + "+".join(reason) if reason else "detecting"
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