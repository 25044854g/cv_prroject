import cv2
import numpy as np

# 打开摄像头
cap = cv2.VideoCapture(0)

print("手部检测已启动（OpenCV版本）")
print(" 请在摄像头前挥动你的手")
print("按 'q' 键退出...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 转换为HSV（更好检测皮肤颜色）
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 定义皮肤颜色范围（HSV）
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # 创建皮肤颜色掩膜
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # 进行形态学操作来清理噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 获取最大的轮廓（应该是手）
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 在画面上绘制轮廓
        cv2.drawContours(frame, [largest_contour], 0, (0, 255, 0), 2)
        
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # 显示面积信息
        area = cv2.contourArea(largest_contour)
        cv2.putText(frame, f"Hand Area: {area}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 同时显示掩膜（方便调试）
    cv2.imshow('Hand Detection', frame)
    cv2.imshow('Mask', mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("已退出")
        break

cap.release()
cv2.destroyAllWindows()