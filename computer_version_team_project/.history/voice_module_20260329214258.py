"""
Voice Recognition Module
语音输入目标物体，使用 OpenAI Whisper 本地识别 + 语音确认
"""

import whisper
import numpy as np
import pyttsx3
import pyaudio
import wave
import tempfile
import os

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
    """语音识别模块：通过 Whisper 语音输入并确认目标物体"""

    def __init__(self, yolo_class_names, default_target="cell phone", whisper_model="base"):
        """
        Args:
            yolo_class_names: YOLO 模型所有类名列表 (list of str)
            default_target: 默认目标物体名称
            whisper_model: Whisper 模型大小 ("tiny", "base", "small", "medium")
        """
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model(whisper_model)
        print(f"Whisper '{whisper_model}' model loaded")
        
        self.yolo_class_names = [name.lower() for name in yolo_class_names]
        self.default_target = default_target
        self.max_retries = 5
        
        # 录音参数
        self.RATE = 16000
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        self.CHUNK = 1024

    def speak(self, text):
        """语音输出，每次重新初始化引擎避免卡死"""
        print(f"[VOICE] {text}")
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    def record_audio(self, duration=5):
        """录音指定秒数，返回临时 wav 文件路径"""
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT, channels=self.CHANNELS,
                        rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
        
        print(f"Recording for {duration} seconds...")
        frames = []
        for _ in range(0, int(self.RATE / self.CHUNK * duration)):
            data = stream.read(self.CHUNK, exception_on_overflow=False)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # 保存为临时 wav 文件
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        wf = wave.open(tmp.name, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        return tmp.name

    def listen_once(self, prompt_text, duration=5):
        """语音提示 + 录音 + Whisper 识别，返回文本或 None"""
        self.speak(prompt_text)
        try:
            wav_path = self.record_audio(duration=duration)
            result = self.whisper_model.transcribe(wav_path, language="en", fp16=False)
            text = result["text"].strip().lower()
            # 清理临时文件
            os.unlink(wav_path)
            if text:
                print(f"You said: {text}")
                return text
            else:
                print("Could not hear clearly.")
                return None
        except Exception as e:
            print(f"Recognition error: {e}")
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
            # Step 1: 听用户说想找什么 (录5秒)
            speech_text = self.listen_once("What object do you want to find? Please speak now.", duration=5)
            if speech_text is None:
                continue

            # Step 2: 提取关键词
            extracted = self.extract_object_from_speech(speech_text)
            print(f"Extracted target: {extracted}")

            # Step 3: 语音询问确认 (录3秒，只需说yes/no)
            confirm_text = self.listen_once(f"Do you want to find {extracted}? Please say yes or no.", duration=3)
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
