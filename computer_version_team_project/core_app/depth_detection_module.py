import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class DepthDetector:
    """
    Depth detection module using stereo vision approximation
    Estimates depth relationships between hand and object
    """
    
    def __init__(self, target_object="cell phone"):
        self.target_object = target_object.lower()
        self.yolo_model = YOLO('yolov8m.pt')
        self.prev_bboxes = {}
        self.frame_width = None
        self.frame_height = None
        
    def process_frame(self, frame):
        """
        Process frame for depth information
        
        Returns:
            Dictionary with depth analysis results
        """
        self.frame_height, self.frame_width = frame.shape[:2]
        
        results = self.yolo_model(frame, conf=0.5)
        
        hand_bbox = None
        object_bbox = None
        
        for r in results:
            for cls_id, conf in zip(r.boxes.cls, r.boxes.conf):
                cls_name = self.yolo_model.names[int(cls_id)].lower()
                
                if 'hand' in cls_name:
                    bbox = r.boxes.xyxy[int(cls_id)]
                    hand_bbox = [int(x) for x in bbox]
                    
                elif cls_name == self.target_object:
                    bbox = r.boxes.xyxy[len(r.boxes.cls) - 1]
                    object_bbox = [int(x) for x in bbox]
        
        depth_info = {
            'is_same_depth': True,
            'depth_diff': 0,
            'hand_bbox': hand_bbox,
            'object_bbox': object_bbox,
            'confidence': 1.0
        }
        
        if hand_bbox and object_bbox:
            hand_y_center = (hand_bbox[1] + hand_bbox[3]) // 2
            object_y_center = (object_bbox[1] + object_bbox[3]) // 2
            
            y_diff = abs(hand_y_center - object_y_center)
            
            if y_diff > 50:
                depth_info['is_same_depth'] = False
                depth_info['depth_diff'] = hand_y_center - object_y_center
                depth_info['confidence'] = max(0.5, 1.0 - y_diff / 100)
            else:
                depth_info['is_same_depth'] = True
                depth_info['depth_diff'] = 0
                depth_info['confidence'] = min(1.0, 1.0 - y_diff / 100)
        else:
            depth_info['is_same_depth'] = True
            depth_info['confidence'] = 0.0
        
        return depth_info


class DepthDetectorAdvanced:
    """
    Advanced depth detection using optical flow
    """
    
    def __init__(self, target_object="cell phone"):
        self.target_object = target_object.lower()
        self.yolo_model = YOLO('yolov8m.pt')
        self.prev_gray = None
        self.prev_hand_bbox = None
        
    def process_frame(self, frame):
        """
        Process frame using optical flow
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        results = self.yolo_model(frame, conf=0.5)
        
        hand_bbox = None
        object_bbox = None
        
        for r in results:
            for i, (cls_id, conf) in enumerate(zip(r.boxes.cls, r.boxes.conf)):
                cls_name = self.yolo_model.names[int(cls_id)].lower()
                
                if 'hand' in cls_name:
                    bbox = r.boxes.xyxy[i]
                    hand_bbox = [int(x) for x in bbox]
                    
                elif cls_name == self.target_object:
                    bbox = r.boxes.xyxy[i]
                    object_bbox = [int(x) for x in bbox]
        
        depth_info = {
            'is_same_depth': True,
            'depth_diff': 0,
            'hand_bbox': hand_bbox,
            'object_bbox': object_bbox,
            'optical_flow_magnitude': 0,
            'confidence': 1.0
        }
        
        if hand_bbox and self.prev_hand_bbox:
            flow_x = hand_bbox[0] - self.prev_hand_bbox[0]
            flow_y = hand_bbox[1] - self.prev_hand_bbox[1]
            magnitude = np.sqrt(flow_x**2 + flow_y**2)
            depth_info['optical_flow_magnitude'] = magnitude
        
        if hand_bbox and object_bbox:
            hand_center_y = (hand_bbox[1] + hand_bbox[3]) // 2
            object_center_y = (object_bbox[1] + object_bbox[3]) // 2
            
            y_diff = abs(hand_center_y - object_center_y)
            depth_info['depth_diff'] = y_diff
            
            if y_diff < 40:
                depth_info['is_same_depth'] = True
                depth_info['confidence'] = 1.0 - y_diff / 100
            else:
                depth_info['is_same_depth'] = False
                depth_info['confidence'] = max(0.3, 1.0 - y_diff / 150)
        
        self.prev_hand_bbox = hand_bbox
        self.prev_gray = gray
        
        return depth_info
