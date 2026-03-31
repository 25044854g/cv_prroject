import cv2

print("打开摄像头并画矩形...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 无法打开摄像头")
else:
    print("✓ 摄像头已打开")
    print("按 'q' 键退出...")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 获取图片的尺寸
        height, width = frame.shape[:2]
        
        # 在图像上画一个矩形框
        # 左上角 (100, 100)，右下角 (300, 300)
        x1, y1 = 100, 100
        x2, y2 = 300, 300
        
        # 画蓝色矩形框（颜色是BGR格式：B蓝, G绿, R红）
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 还可以试试其他颜色：
        # 绿色：(0, 255, 0)
        # 红色：(0, 0, 255)
        # 黄色：(0, 255, 255)
        
        # 也可以在多个地方画框
        cv2.rectangle(frame, (400, 200), (600, 400), (0, 255, 0), 2)  # 绿色
        
        cv2.imshow('Camera with Rectangle', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("已退出")
            break
    
    cap.release()
    cv2.destroyAllWindows()