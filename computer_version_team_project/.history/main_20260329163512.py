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
    annotated_frame = hand_detector.process_frame(frame)
    
    # Process frame with depth detector
    depth_result = depth_detector.process_frame(frame)
    
    # 合并两个结果（使用深度检测的结果，因为它包含更多信息）
    final_frame = depth_result['annotated_frame']
    
    # Display
    if frame_count % 3 == 0:
        cv2.imshow('Hand-Object Detection with Depth', final_frame)
    
    if cv2.waitKey(500) & 0xFF == ord('q'):
        print("Exited")
        break

cap.release()
cv2.destroyAllWindows()