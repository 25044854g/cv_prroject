"""
Main application file
Imports and uses the HandObjectDetector module
"""

import cv2
from hand_detection_module import HandObjectDetector


def main():
    """Main application loop"""
    
    # Initialize detector
    detector = HandObjectDetector()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return
    
    print("✓ Camera opened")
    print("Point your hand toward an object (NOT a person)")
    print("Press 'q' to exit...\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame with detector
        result = detector.process_frame(frame)
        annotated_frame = result['annotated_frame']
        
        # Display object info
        if result['object_name'] is not None:
            cv2.putText(annotated_frame, 
                       f"Object: {result['object_name']} ({result['object_confidence']:.1%})", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.putText(annotated_frame, 
                       f"Distance: {result['distance']} px", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Display direction
            if result['direction'] is not None:
                cv2.putText(annotated_frame, result['direction'], 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, result['direction_color'], 2)
            
            # Display instruction
            if result['instruction'] is not None:
                cv2.putText(annotated_frame, result['instruction'], 
                           (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, result['direction_color'], 3)
            
            # Display grab alert
            if result['is_ready_to_grab']:
                cv2.putText(annotated_frame, "READY TO GRAB!", 
                           (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.rectangle(annotated_frame, (5, 185), (350, 220), (0, 255, 0), 3)
        else:
            if result['hand_x'] is None:
                cv2.putText(annotated_frame, "No hand detected", 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Display
        if frame_count % 3 == 0:
            cv2.imshow('Hand-Object Detection', annotated_frame)
        
        # Exit on 'q'
        if cv2.waitKey(500) & 0xFF == ord('q'):
            print("Exited")
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()