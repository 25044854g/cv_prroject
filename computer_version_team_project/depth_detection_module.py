import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os


class DepthDetector:
    """深度检测模块"""
    
    def __init__(self, target_object="cell phone"):
        """初始化深度检测模块"""
        # 获取路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir
        
        # 初始化 YOLO
        self.yolo_model = YOLO(os.path.join(base_dir, 'yolov8m.pt'))
        self.hand_backend = 'mediapipe'
        self.detector = None
        self.pose_model = None

        # 初始化 MediaPipe 手部检测，若被系统策略阻止则回退到 YOLO pose。
        model_path = os.path.join(base_dir, 'hand_landmarker.task')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find {model_path} file")

        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
            self.detector = vision.HandLandmarker.create_from_options(options)
        except OSError as error:
            self.hand_backend = 'yolo_pose'
            self.pose_model = YOLO(os.path.join(base_dir, 'yolov8n-pose.pt'))
            print(f"MediaPipe depth backend unavailable ({error}). Falling back to YOLO pose.")
        
        self.DEPTH_THRESHOLD = 0.05
        self.target_object = target_object.lower()

    def get_hand_anchor_point(self, hand_landmarks, frame_width, frame_height):
        """与可运行示例保持一致，使用中指指尖作为手部位置。"""
        middle_finger = hand_landmarks[12]
        anchor_x = int(middle_finger.x * frame_width)
        anchor_y = int(middle_finger.y * frame_height)
        return anchor_x, anchor_y

    
    def estimate_hand_depth(self, hand_landmarks, frame_width, frame_height):
        """
        估计手的深度
        基于手的大小：手越大 = 离摄像头越近
        返回 0-1，1 表示离摄像头最近
        """
        hand_x_coords = [lm.x for lm in hand_landmarks]
        hand_y_coords = [lm.y for lm in hand_landmarks]
        
        hand_x_min = min(hand_x_coords)
        hand_x_max = max(hand_x_coords)
        hand_y_min = min(hand_y_coords)
        hand_y_max = max(hand_y_coords)
        
        hand_width = hand_x_max - hand_x_min
        hand_height = hand_y_max - hand_y_min
        hand_size = (hand_width + hand_height) / 2
        
        # 将大小转换为深度值 (0-1)
        hand_depth = min(1.0, hand_size * 2)
        
        return hand_depth, hand_size
    
    def estimate_object_depth(self, box, frame_width, frame_height):
        """
        估计物体的深度
        基于物体框的大小：框越大 = 离摄像头越近
        返回 0-1，1 表示离摄像头最近
        """
        x1, y1, x2, y2 = box.xyxy[0]
        
        obj_width = (x2 - x1) / frame_width
        obj_height = (y2 - y1) / frame_height
        obj_size = (obj_width + obj_height) / 2
        
        # 将大小转换为深度值 (0-1)
        obj_depth = min(1.0, obj_size * 2)
        
        return obj_depth, obj_size

    def detect_hand_with_pose(self, frame, annotated_frame):
        """Use pose wrists as a fallback hand position when MediaPipe is blocked."""
        hand_detected = False
        hand_x, hand_y = None, None
        pose_results = self.pose_model(frame, verbose=False)
        keypoints = pose_results[0].keypoints

        if keypoints is not None and keypoints.xy is not None and len(keypoints.xy) > 0:
            person_points = keypoints.xy[0]
            wrist_points = []

            for index in (9, 10):
                wrist_x, wrist_y = person_points[index]
                if wrist_x > 0 and wrist_y > 0:
                    wrist_points.append((int(wrist_x), int(wrist_y)))

            if wrist_points:
                hand_detected = True
                hand_x = int(sum(point[0] for point in wrist_points) / len(wrist_points))
                hand_y = int(sum(point[1] for point in wrist_points) / len(wrist_points))

                for wrist_x, wrist_y in wrist_points:
                    cv2.circle(annotated_frame, (wrist_x, wrist_y), 8, (0, 165, 255), -1)

                cv2.circle(annotated_frame, (hand_x, hand_y), 12, (0, 0, 255), -1)
                cv2.putText(annotated_frame, "HAND (POSE)", (hand_x - 70, hand_y - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return hand_detected, hand_x, hand_y
    
    def process_frame(self, frame):
        """
        处理单帧，进行深度检测
        
        Args:
            frame: 输入帧
            
        Returns:
            Dictionary 包含:
            - annotated_frame: 标注后的帧
            - hand_detected: 手是否检测到
            - target_detected: 目标物体是否检测到
            - hand_depth: 手的深度值
            - target_depth: 目标物体的深度值
            - hand_size: 手的大小
            - target_size: 目标物体的大小
            - depth_diff: 深度差异
            - is_same_depth: 是否深度相同
        """
        height, width, c = frame.shape
        annotated_frame = frame.copy()
        
        # 1. 目标物体检测
        results_yolo = self.yolo_model(frame)
        boxes = results_yolo[0].boxes
        
        target_detected = False
        target_x, target_y = None, None
        target_depth = None
        target_size = None
        
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = self.yolo_model.names[class_id]
            
            if class_name.lower() == self.target_object:
                target_detected = True
                x1, y1, x2, y2 = box.xyxy[0]
                target_x = (int(x1) + int(x2)) // 2
                target_y = (int(y1) + int(y2)) // 2
                
                # 估计目标物体深度
                target_depth, target_size = self.estimate_object_depth(box, width, height)
                
                # 绘制目标物体框（蓝色）
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                             (255, 0, 0), 3)
                cv2.circle(annotated_frame, (target_x, target_y), 8, (255, 0, 0), -1)
                break
        
        # 2. 手部检测
        hand_detected = False
        hand_x, hand_y = None, None
        hand_depth = None
        hand_size = None
        
        if self.hand_backend == 'mediapipe':
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.detector.detect(mp_image)

            if detection_result.hand_landmarks:
                hand_detected = True
                for hand_landmarks in detection_result.hand_landmarks:
                    hand_x, hand_y = self.get_hand_anchor_point(hand_landmarks, width, height)
                    hand_depth, hand_size = self.estimate_hand_depth(hand_landmarks, width, height)

                    connections = [
                        (0, 1), (1, 2), (2, 3), (3, 4),
                        (0, 5), (5, 6), (6, 7), (7, 8),
                        (0, 9), (9, 10), (10, 11), (11, 12),
                        (0, 13), (13, 14), (14, 15), (15, 16),
                        (0, 17), (17, 18), (18, 19), (19, 20),
                        (5, 9), (9, 13), (13, 17)
                    ]

                    for landmark in hand_landmarks:
                        lx = int(landmark.x * width)
                        ly = int(landmark.y * height)
                        cv2.circle(annotated_frame, (lx, ly), 4, (0, 255, 0), -1)

                    for start, end in connections:
                        start_point = (int(hand_landmarks[start].x * width),
                                      int(hand_landmarks[start].y * height))
                        end_point = (int(hand_landmarks[end].x * width),
                                    int(hand_landmarks[end].y * height))
                        cv2.line(annotated_frame, start_point, end_point, (0, 255, 0), 2)

                    cv2.circle(annotated_frame, (hand_x, hand_y), 12, (0, 0, 255), -1)
        else:
            hand_detected, hand_x, hand_y = self.detect_hand_with_pose(frame, annotated_frame)
        
        # 3. 添加文本信息
        start_y = 40
        
        # 标题
        cv2.putText(annotated_frame, "=== Depth Detection Test ===", 
                   (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        start_y += 40
        
        # 手部信息
        if hand_detected and hand_depth is not None:
            cv2.putText(annotated_frame, f"Hand Size: {hand_size:.4f}", 
                       (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Hand Depth: {hand_depth:.4f}", 
                       (10, start_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            start_y += 75
        elif hand_detected and self.hand_backend != 'mediapipe':
            cv2.putText(annotated_frame, "Hand: detected by YOLO pose", 
                       (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(annotated_frame, "Depth: unavailable in fallback mode", 
                       (10, start_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            start_y += 75
        else:
            cv2.putText(annotated_frame, "Hand: Not detected", 
                       (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
            start_y += 40
        
        # 目标物体信息
        if target_detected and target_depth is not None:
            cv2.putText(annotated_frame, f"Target Size: {target_size:.4f}", 
                       (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.putText(annotated_frame, f"Target Depth: {target_depth:.4f}", 
                       (10, start_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            start_y += 75
        else:
            cv2.putText(annotated_frame, "Target: Not detected", 
                       (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
            start_y += 40
        
        # 深度对比结果
        cv2.putText(annotated_frame, "--- Result ---", 
                   (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        start_y += 40
        
        depth_diff = None
        is_same_depth = False
        
        if hand_detected and target_detected and hand_depth is not None and target_depth is not None:
            depth_diff = hand_depth - target_depth
            
            cv2.putText(annotated_frame, f"Depth Difference: {depth_diff:.4f}", 
                       (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 0), 2)
            start_y += 40
            
            # 判断是否在同一水平线
            if abs(depth_diff) < self.DEPTH_THRESHOLD:
                # 成功：在同一水平线
                is_same_depth = True
                cv2.putText(annotated_frame, "✓ SUCCESS!", 
                           (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(annotated_frame, "Same depth level!", 
                           (10, start_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 绘制成功指示框
                cv2.rectangle(annotated_frame, (5, start_y - 35), (320, start_y + 55), (0, 255, 0), 3)
            else:
                # 失败：不在同一水平线
                if depth_diff > 0:
                    status = "Hand CLOSER"
                    color = (0, 165, 255)  # 橙色
                else:
                    status = "Hand FARTHER"
                    color = (0, 0, 255)  # 红色
                
                cv2.putText(annotated_frame, "✗ DIFFERENT DEPTH", 
                           (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                cv2.putText(annotated_frame, status, 
                           (10, start_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            cv2.putText(annotated_frame, "Waiting for detection...", 
                       (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
        
        return {
            'annotated_frame': annotated_frame,
            'hand_detected': hand_detected,
            'target_detected': target_detected,
            'hand_depth': hand_depth,
            'target_depth': target_depth,
            'hand_size': hand_size,
            'target_size': target_size,
            'depth_diff': depth_diff,
            'is_same_depth': is_same_depth
        }