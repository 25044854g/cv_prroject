import cv2
import mediapipe as mp

# 初始化 MediaPipe 手部检测
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # 不是静态图像，是视频
    max_num_hands=2,          # 最多检测2只手
    min_detection_confidence=0.5  # 检测置信度
)

mp_drawing = mp.solutions.drawing_utils

# 打开摄像头
cap = cv2.VideoCapture(0)

print("✓ 手部检测已启动")
print("👋 请在摄像头前挥动你的手")
print("按 'q' 键退出...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 获取图片尺寸
    height, width, c = frame.shape
    
    # 将BGR图像转换为RGB（MediaPipe需要RGB格式）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 进行手部检测
    results = hands.process(rgb_frame)
    
    # 如果检测到手，就画出来
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 使用 MediaPipe 的绘制工具画出手部关键点和连线
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
    
    cv2.imshow('Hand Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("已退出")
        break

cap.release()
cv2.destroyAllWindows()