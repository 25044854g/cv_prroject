import cv2

# 操作1：读取一张图片
img = cv2.imread('photo.jpg')
cv2.imshow('My Photo', img)  # 在窗口里显示
cv2.waitKey(0)  # 按任意键关闭

# 操作2：读取摄像头实时画面（很重要！）
cap = cv2.VideoCapture(0)  # 0表示默认摄像头
while True:
    ret, frame = cap.read()  # 每一帧都是一张图片
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按'q'退出
        break
cap.release()
cv2.destroyAllWindows()

# 操作3：在图像上画矩形框
cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 画蓝色框