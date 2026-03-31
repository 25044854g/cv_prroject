import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # 在画面中央画一个圆
    cv2.circle(frame, (frame.shape[1]//2, frame.shape[0]//2), 50, (0, 255, 0), 2)
    
    cv2.imshow('My First Program', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()