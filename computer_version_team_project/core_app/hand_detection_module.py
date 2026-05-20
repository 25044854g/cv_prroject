import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandObjectDetector:
    """
    Hand and object detection using MediaPipe + YOLO
    Detects hand keypoints and object bounding boxes
    """
    
    def __init__(self, target_object="cell phone"):
        self.target_object = target_object.lower()
        self.yolo_model = YOLO('yolov8m.pt')
        
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.frame_width = None
        self.frame_height = None
        self.target_bbox = None
        self.hand_landmarks = None
        self.hand_detection_confidence = 0.0
        
    def process_frame(self, frame):
        """
        Process frame with hand and object detection
        
        Returns:
            Dictionary with detection results
        """
        self.frame_height, self.frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results_hands = self.hands.process(rgb_frame)
        self.hand_landmarks = None
        
        if results_hands.multi_hand_landmarks and len(results_hands.multi_hand_landmarks) > 0:
            self.hand_landmarks = results_hands.multi_hand_landmarks[0]
            self.hand_detection_confidence = (
                results_hands.multi_handedness[0].classification[0].score
                if results_hands.multi_handedness else 0.0
            )
        
        yolo_results = self.yolo_model(frame, conf=0.5)
        
        self.target_bbox = None
        annotated_frame = frame.copy()
        
        for r in yolo_results:
            for i, (cls_id, conf) in enumerate(zip(r.boxes.cls, r.boxes.conf)):
                cls_name = self.yolo_model.names[int(cls_id)].lower()
                
                if cls_name == self.target_object and self.target_bbox is None:
                    bbox = r.boxes.xyxy[i]
                    self.target_bbox = [int(x) for x in bbox]
                    
                    cv2.rectangle(annotated_frame, 
                                (self.target_bbox[0], self.target_bbox[1]),
                                (self.target_bbox[2], self.target_bbox[3]),
                                (0, 255, 0), 2)
                    
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(annotated_frame, label,
                              (self.target_bbox[0], self.target_bbox[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        if self.hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                self.hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS
            )
        
        detection_result = self._compute_detection_result()
        detection_result['annotated_frame'] = annotated_frame
        
        return detection_result
    
    def _compute_detection_result(self):
        """Compute all detection metrics"""
        result = {
            'hand_detected': self.hand_landmarks is not None,
            'target_detected': self.target_bbox is not None,
            'hand_confidence': self.hand_detection_confidence,
            'distance_2d': None,
            'direction': None,
            'is_vertically_aligned': False,
            'is_horizontally_aligned': False,
            'hand_bbox': None,
            'target_bbox': self.target_bbox,
        }
        
        if not self.hand_landmarks or not self.target_bbox:
            return result
        
        hand_center = self._get_hand_center()
        target_center = [
            (self.target_bbox[0] + self.target_bbox[2]) // 2,
            (self.target_bbox[1] + self.target_bbox[3]) // 2
        ]
        
        dx = target_center[0] - hand_center[0]
        dy = target_center[1] - hand_center[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        result['distance_2d'] = int(distance)
        
        if dx < -30:
            result['direction'] = 'LEFT'
        elif dx > 30:
            result['direction'] = 'RIGHT'
        else:
            result['direction'] = 'CENTER_X'
        
        if dy < -30:
            direction_y = 'ABOVE'
        elif dy > 30:
            direction_y = 'BELOW'
        else:
            direction_y = 'CENTER_Y'
        
        if result['direction'] != 'CENTER_X':
            result['direction'] += f" {direction_y}"
        else:
            result['direction'] = direction_y
        
        result['is_vertically_aligned'] = abs(dy) < 30
        result['is_horizontally_aligned'] = abs(dx) < 30
        
        hand_bbox = self._get_hand_bounding_box()
        result['hand_bbox'] = hand_bbox
        
        return result
    
    def _get_hand_center(self):
        """Get center of hand from landmarks"""
        if not self.hand_landmarks:
            return [self.frame_width // 2, self.frame_height // 2]
        
        x_coords = [lm.x * self.frame_width for lm in self.hand_landmarks.landmark]
        y_coords = [lm.y * self.frame_height for lm in self.hand_landmarks.landmark]
        
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        
        return [center_x, center_y]
    
    def _get_hand_bounding_box(self):
        """Get bounding box of hand from landmarks"""
        if not self.hand_landmarks:
            return None
        
        x_coords = [lm.x * self.frame_width for lm in self.hand_landmarks.landmark]
        y_coords = [lm.y * self.frame_height for lm in self.hand_landmarks.landmark]
        
        x_min = int(min(x_coords))
        y_min = int(min(y_coords))
        x_max = int(max(x_coords))
        y_max = int(max(y_coords))
        
        return [x_min, y_min, x_max, y_max]


class HandDetectorFallback:
    """
    Fallback hand detection using YOLO-Pose when MediaPipe fails
    """
    
    def __init__(self, target_object="cell phone"):
        self.target_object = target_object.lower()
        try:
            self.pose_model = YOLO('yolov8n-pose.pt')
        except:
            self.pose_model = None
        
        self.yolo_model = YOLO('yolov8m.pt')
        self.frame_width = None
        self.frame_height = None
        
    def detect_hand_with_pose(self, frame):
        """Detect hand using YOLO-Pose as fallback"""
        if not self.pose_model:
            return None
        
        pose_results = self.pose_model(frame)
        
        if len(pose_results) > 0 and pose_results[0].keypoints:
            keypoints = pose_results[0].keypoints
            
            if len(keypoints) > 10:
                wrist = keypoints[9]
                shoulder = keypoints[5]
                
                if wrist[2] > 0.5 and shoulder[2] > 0.5:
                    x_coords = [kp[0] for kp in keypoints[5:11] if kp[2] > 0.5]
                    y_coords = [kp[1] for kp in keypoints[5:11] if kp[2] > 0.5]
                    
                    if x_coords and y_coords:
                        return {
                            'hand_detected': True,
                            'center': [np.mean(x_coords), np.mean(y_coords)],
                            'confidence': float(np.mean([kp[2] for kp in keypoints[5:11]]))
                        }
        
        return None
    
    def process_frame(self, frame):
        """Process with fallback to pose if needed"""
        self.frame_height, self.frame_width = frame.shape[:2]
        
        hand_result = self.detect_hand_with_pose(frame)
        
        if hand_result:
            return {
                'hand_detected': True,
                'target_detected': False,
                'hand_confidence': hand_result['confidence'],
                'hand_center': hand_result['center'],
                'distance_2d': None,
                'method': 'pose',
                'annotated_frame': frame
            }
        
        return {
            'hand_detected': False,
            'target_detected': False,
            'hand_confidence': 0.0,
            'method': 'none',
            'annotated_frame': frame
        }
