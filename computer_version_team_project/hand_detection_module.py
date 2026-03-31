import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os


class HandObjectDetector:
    """Hand and Object Detection Module"""
    
    def __init__(self, target_object="cell phone"):
        """Initialize the detector with models"""
        # Resolve paths relative to this script's directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Initialize YOLO with Medium model for better accuracy
        self.yolo_model = YOLO(os.path.join(base_dir, 'yolov8m.pt'))
        
        # Initialize MediaPipe hand detection
        model_path = os.path.join(base_dir, 'hand_landmarker.task')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find {model_path} file")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.PERSON_CLASS_ID = 0
        self.target_object = target_object.lower()

    def get_hand_anchor_point(self, hand_landmarks, frame_width, frame_height):
        """Estimate a stable hand position from the palm instead of fingertips."""
        anchor_indices = [0, 5, 9, 13, 17]
        anchor_x = int(sum(hand_landmarks[index].x for index in anchor_indices) / len(anchor_indices) * frame_width)
        anchor_y = int(sum(hand_landmarks[index].y for index in anchor_indices) / len(anchor_indices) * frame_height)
        return anchor_x, anchor_y

    
    def estimate_hand_depth(self, hand_landmarks, frame_width, frame_height):
        """估计手的深度"""
        hand_x_coords = [lm.x for lm in hand_landmarks]
        hand_y_coords = [lm.y for lm in hand_landmarks]
        
        hand_x_min = min(hand_x_coords)
        hand_x_max = max(hand_x_coords)
        hand_y_min = min(hand_y_coords)
        hand_y_max = max(hand_y_coords)
        
        hand_width = hand_x_max - hand_x_min
        hand_height = hand_y_max - hand_y_min
        hand_size = (hand_width + hand_height) / 2
        
        hand_depth = min(1.0, hand_size * 2)
        
        return hand_depth, hand_size
    
    def estimate_object_depth(self, box, frame_width, frame_height):
        """估计物体的深度"""
        x1, y1, x2, y2 = box.xyxy[0]
        
        obj_width = (x2 - x1) / frame_width
        obj_height = (y2 - y1) / frame_height
        obj_size = (obj_width + obj_height) / 2
        
        obj_depth = min(1.0, obj_size * 2)
        
        return obj_depth, obj_size
    
    def is_hand_in_front_of_object(self, hand_depth_info, object_depth_info):
        """判断手是否在物体前面"""
        hand_depth = hand_depth_info['depth']
        object_depth = object_depth_info['depth']
        
        depth_diff = hand_depth - object_depth
        
        DEPTH_THRESHOLD = 0.1
        is_in_front = depth_diff > DEPTH_THRESHOLD
        
        return {
            'depth_diff': depth_diff,
            'is_in_front': is_in_front,
            'hand_depth': hand_depth,
            'object_depth': object_depth
        }

    def process_frame(self, frame):
        """Process a single frame and return both annotated frame and detection info"""
        height, width, c = frame.shape

        results_yolo = self.yolo_model(frame)
        annotated_frame = results_yolo[0].plot()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)

        # Get hand position
        hand_x, hand_y = None, None
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                hand_x, hand_y = self.get_hand_anchor_point(hand_landmarks, width, height)

                # Draw all landmarks (green dots)
                for landmark in hand_landmarks:
                    lx = int(landmark.x * width)
                    ly = int(landmark.y * height)
                    cv2.circle(annotated_frame, (lx, ly), 4, (0, 255, 0), -1)

                # Draw hand skeleton
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

                # Draw red circle at palm center
                cv2.circle(annotated_frame, (hand_x, hand_y), 12, (0, 0, 255), -1)
                cv2.putText(annotated_frame, "HAND", (hand_x - 50, hand_y - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 3. Analyze hand-object relationship
        # Only detect target object
        boxes = results_yolo[0].boxes
        target_box = None

        if hand_x is not None and len(boxes) > 0:
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = self.yolo_model.names[class_id]

                if class_name.lower() == self.target_object:
                    target_box = box
                    break

        # 初始化返回信息
        result_info = {
            'annotated_frame': annotated_frame,
            'distance_2d': None,
            'direction': None,
            'is_vertically_aligned': False,
            'is_horizontally_aligned': False,
            'is_in_front': False
        }

        # Process the target object (if found and not a person)
        if hand_x is not None and target_box is not None:
            box = target_box

            x1, y1, x2, y2 = box.xyxy[0]
            obj_x = (int(x1) + int(x2)) // 2
            obj_y = (int(y1) + int(y2)) // 2

            class_id = int(box.cls[0])
            class_name = self.yolo_model.names[class_id]
            confidence = float(box.conf[0])

            # Draw target box with different color (magenta for non-person targets)
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                         (255, 0, 255), 3)

            # Calculate distance
            dx = hand_x - obj_x
            dy = hand_y - obj_y
            distance_2d = int(np.sqrt(dx**2 + dy**2))

            # Draw line between hand and object
            cv2.line(annotated_frame, (hand_x, hand_y), (obj_x, obj_y), (255, 255, 0), 3)

            # Display object info
            cv2.putText(annotated_frame, f"Object: {class_name} ({confidence:.1%})",
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(annotated_frame, f"Distance: {distance_2d} px",
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Determine alignment independently for each axis
            threshold = 80
            is_h_aligned = abs(dx) <= threshold
            is_v_aligned = abs(dy) <= threshold

            # Build direction/instruction text
            directions = []
            instructions = []
            color = (0, 255, 0)  # default green

            if not is_h_aligned:
                color = (0, 165, 255)
                if dx < 0:
                    directions.append("Hand is RIGHT of object")
                    instructions.append("Move LEFT")
                else:
                    directions.append("Hand is LEFT of object")
                    instructions.append("Move RIGHT")

            if not is_v_aligned:
                color = (0, 165, 255)
                if dy < 0:
                    directions.append("Hand is ABOVE object")
                    instructions.append("Move DOWN")
                else:
                    directions.append("Hand is BELOW object")
                    instructions.append("Move UP")

            if is_h_aligned and is_v_aligned:
                directions.append("Aligned")
                color = (0, 255, 0)

            direction_text = " | ".join(directions)
            instruction_text = " | ".join(instructions)

            cv2.putText(annotated_frame, direction_text,
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(annotated_frame, instruction_text,
                       (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            # 更新返回信息
            result_info['distance_2d'] = distance_2d
            result_info['direction'] = direction_text
            result_info['is_vertically_aligned'] = is_v_aligned
            result_info['is_horizontally_aligned'] = is_h_aligned
        else:
            if hand_x is None:
                cv2.putText(annotated_frame, "No hand detected",
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        result_info['annotated_frame'] = annotated_frame
        return result_info