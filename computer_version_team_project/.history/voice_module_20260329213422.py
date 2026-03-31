"""
Voice Recognition Module
语音输入目标物体，支持自然语言提取关键词 + 语音确认
"""

import speech_recognition as sr
import pyttsx3

# 常见别名映射到 YOLO COCO 类名
ALIAS_MAP = {
    "phone": "cell phone",
    "cellphone": "cell phone",
    "mobile": "cell phone",
    "mobile phone": "cell phone",
    "smartphone": "cell phone",
    "tv": "tv",
    "television": "tv",
    "monitor": "tv",
    "screen": "tv",
    "couch": "couch",
    "sofa": "couch",
    "bike": "bicycle",
    "motorbike": "motorcycle",
    "aeroplane": "airplane",
    "plane": "airplane",
    "remote control": "remote",
    "glasses": "wine glass",
    "glass": "wine glass",
    "mouse": "mouse",
    "computer mouse": "mouse",
    "mug": "cup",
}


class VoiceDetector:
    """语音识别模块：通过语音输入并确认目标物体"""

    def __init__(self, yolo_class_names, default_target="cell phone"):
        """
        Args:
            yolo_class_names: YOLO 模型所有类名列表 (list of str)
            default_target: 默认目标物体名称
        """
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.yolo_class_names = [name.lower() for name in yolo_class_names]
        self.default_target = default_target
        self.max_retries = 5

    def speak(self, text):
        """语音输出，每次重新初始化引擎避免卡死"""
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    def listen_once(self, prompt_text):
        """语音提示 + 监听一次语音输入，返回识别文本或 None"""
        print(prompt_text)
        self.speak(prompt_text)
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=15, phrase_time_limit=10)
            text = self.recognizer.recognize_google(audio).lower()
            print(f"You said: {text}")
            return text
        except (sr.WaitTimeoutError, sr.UnknownValueError):
            print("Could not hear clearly.")
            return None
        except sr.RequestError as e:
            print(f"Speech service error: {e}")
            return None

    def extract_object_from_speech(self, speech_text):
        """从自然语言中提取 YOLO 可识别的物体名称"""
        words = speech_text.lower().strip()

        # 1. 先检查别名（优先匹配长别名）
        for alias in sorted(ALIAS_MAP.keys(), key=len, reverse=True):
            if alias in words:
                return ALIAS_MAP[alias]

        # 2. 检查 YOLO 类名（优先匹配多词类名）
        for class_name in sorted(self.yolo_class_names, key=len, reverse=True):
            if class_name in words:
                return class_name

        # 3. 逐词匹配
        for word in words.split():
            if word in self.yolo_class_names:
                return word

        # 4. 没匹配到，返回原文
        return words

    def get_target_object(self):
        """
        完整的语音交互流程：询问 -> 提取 -> 确认 -> 返回目标物体名称
        
        Returns:
            str: 确认后的目标物体名称
        """
        print("=" * 50)
        print("Tell me what object you want to find")
        print("Example: 'I want to find my cup'")
        print("=" * 50)

        for attempt in range(self.max_retries):
            # Step 1: 听用户说想找什么
            speech_text = self.listen_once("What object do you want to find? Please speak now.")
            if speech_text is None:
                continue

            # Step 2: 提取关键词
            extracted = self.extract_object_from_speech(speech_text)
            print(f"Extracted target: {extracted}")

            # Step 3: 语音询问确认
            confirm_text = self.listen_once(f"Do you want to find {extracted}? Please say yes or no.")
            if confirm_text is None:
                continue

            # Step 4: 判断回答
            if any(w in confirm_text for w in ["yes", "yeah", "yep", "correct", "right", "sure"]):
                self.speak(f"OK, looking for {extracted}")
                print(f"Confirmed target: {extracted}\n")
                return extracted
            else:
                print("Let's try again.\n")
                self.speak("OK, let's try again.")

        # 超过重试次数
        print(f"Too many retries. Using default target: {self.default_target}")
        self.speak(f"Using default target: {self.default_target}")
        return self.default_target
