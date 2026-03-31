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
import re
from difflib import SequenceMatcher

# 确保 ffmpeg 可用（通过 imageio-ffmpeg 自带的二进制）
try:
    import imageio_ffmpeg
    _ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    _ffmpeg_dir = os.path.dirname(_ffmpeg_exe)
    # 创建一个 ffmpeg.exe 的副本/链接让 Whisper 能找到
    _link_path = os.path.join(_ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(_link_path):
        import shutil
        shutil.copy2(_ffmpeg_exe, _link_path)
    if _ffmpeg_dir not in os.environ["PATH"]:
        os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ["PATH"]
except ImportError:
    pass  # 如果系统已有 ffmpeg 则不需要

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
        self.min_match_score = 0.55
        self.auto_accept_score = 0.72
        self.candidate_display_count = 3

        self.normalized_class_map = {
            self.normalize_text(name): name for name in self.yolo_class_names
        }
        self.normalized_alias_map = {
            self.normalize_text(alias): target for alias, target in ALIAS_MAP.items()
        }
        
        # 录音参数
        self.RATE = 16000
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        self.CHUNK = 1024

    def normalize_text(self, text):
        """Normalize text for robust candidate matching."""
        normalized = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
        return " ".join(normalized.split())

    def score_candidate_against_text(self, normalized_text, candidate_key):
        """Compare a candidate against full text and token windows."""
        if not normalized_text or not candidate_key:
            return 0.0

        if candidate_key == normalized_text:
            return 1.0

        if candidate_key in normalized_text:
            return 0.96

        text_tokens = normalized_text.split()
        candidate_tokens = candidate_key.split()
        window_size = len(candidate_tokens)
        best_score = SequenceMatcher(None, normalized_text, candidate_key).ratio()

        if window_size <= len(text_tokens):
            for start in range(len(text_tokens) - window_size + 1):
                window = " ".join(text_tokens[start:start + window_size])
                best_score = max(best_score, SequenceMatcher(None, window, candidate_key).ratio())

        for token in text_tokens:
            best_score = max(best_score, SequenceMatcher(None, token, candidate_key).ratio())

        return best_score

    def rank_candidate_objects(self, speech_text):
        """Rank known target objects against recognized speech."""
        normalized_text = self.normalize_text(speech_text)
        if not normalized_text:
            return []

        candidate_scores = {}

        for alias_key, canonical_name in self.normalized_alias_map.items():
            score = self.score_candidate_against_text(normalized_text, alias_key)
            candidate_scores[canonical_name] = max(candidate_scores.get(canonical_name, 0.0), score)

        for class_key, canonical_name in self.normalized_class_map.items():
            score = self.score_candidate_against_text(normalized_text, class_key)
            candidate_scores[canonical_name] = max(candidate_scores.get(canonical_name, 0.0), score)

        ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
        return [item for item in ranked if item[1] >= self.min_match_score][:self.candidate_display_count]

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
        words = self.normalize_text(speech_text)

        ranked_candidates = self.rank_candidate_objects(words)
        if ranked_candidates:
            return ranked_candidates[0][0]

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
        print("Say only the object name, for example: cup, mouse, bottle, cell phone")
        print("=" * 50)

        for attempt in range(self.max_retries):
            # Step 1: 只听目标物体名
            speech_text = self.listen_once("Please say only the object name now.", duration=4)
            if speech_text is None:
                continue

            ranked_candidates = self.rank_candidate_objects(speech_text)
            if not ranked_candidates:
                print("Could not confidently match that speech to a supported object.")
                self.speak("I could not match that to a supported object. Please try again.")
                continue

            extracted, score = ranked_candidates[0]
            print(f"Top match: {extracted} ({score:.0%})")

            if score >= self.auto_accept_score:
                self.speak(f"OK, looking for {extracted}")
                print(f"Confirmed target: {extracted}\n")
                return extracted

            preview = ", ".join(
                f"{candidate} ({candidate_score:.0%})" for candidate, candidate_score in ranked_candidates
            )
            print(f"Low-confidence candidates: {preview}")
            self.speak("I am not confident about that object. Please say only the object name again.")

        # 超过重试次数
        print(f"Too many retries. Using default target: {self.default_target}")
        self.speak(f"Using default target: {self.default_target}")
        return self.default_target
