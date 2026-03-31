
目录结构：

computer_version_team_project/
├── main.py                          # 主文件（运行这个）
├── hand_detection_module.py         # 手部检测模块（独立）
├── hand_landmarker.task             # MediaPipe 模型
└── yolov8m.pt                       # YOLO 模型

运行文件，变成模块导入时需要：
作为导入模块需要修改的部分
1. 删除全局 print 输出
删除模块加载时的所有 print 语句
包括 "MediaPipe version"、"Loading MediaPipe model"、"MediaPipe model loaded" 等
这些输出会在导入时立即执行，影响主程序的控制流
2. 删除摄像头初始化代码
删除 cap = cv2.VideoCapture(0)
删除 if not cap.isOpened(): print(...); exit()
摄像头应该由主文件管理，模块不应该关心硬件细节
3. 删除整个主循环
删除 while True: 循环
删除循环内的 ret, frame = cap.read()
删除 frame_count += 1
删除循环内的所有帧处理逻辑
模块只应该处理单个帧，不应该管理循环
4. 删除显示和交互代码
删除 cv2.imshow('Hand-Object Detection (MediaPipe + YOLOv8m)', annotated_frame)
删除 if frame_count % 3 == 0: 的条件显示
删除 cv2.waitKey(500)
删除 if cv2.waitKey(500) & 0xFF == ord('q'): print("Exited"); break
这些都是 UI 层的职责，不是模块的职责

