import cv2
# 操作2：读取摄像头实时画面（很重要！）
cap = cv2.VideoCapture(0)  # 0表示默认摄像头
while True:
    ret, frame = cap.read()  # 每一帧都是一张图片
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按'q'退出
        break
cap.release()
cv2.destroyAllWindows()

