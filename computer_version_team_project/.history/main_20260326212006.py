"""
Main application file
Imports and uses the HandObjectDetector module
"""

import cv2
from hand_detection_module import HandObjectDetector

print("✓ MediaPipe version:", __import__('mediapipe').__version__)
print("Loading MediaPipe model...")

# Initialize detector
detector = HandObjectDetector()

print("✓ MediaPipe model loaded\n")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

print("✓ Hand-Object Detection Started (MediaPipe + YOLOv8m)")
print("Point your hand toward an object (NOT a person)")
print("Press 'q' to exit...\n")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Process frame with detector
    annotated_frame = detector.process_frame(frame)
    
    # Display
    if frame_count % 3 == 0:
        cv2.imshow('Hand-Object Detection (No Person)', annotated_frame)
    
    if cv2.waitKey(500) & 0xFF == ord('q'):
        print("Exited")
        break

cap.release()
cv2.destroyAllWindows()