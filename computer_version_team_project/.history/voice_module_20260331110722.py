"""
Voice Recognition Module
Use Azure Speech SDK to recognize a target object from the microphone.
"""

import os
import re
from difflib import SequenceMatcher

import azure.cognitiveservices.speech as speechsdk
import pyttsx3


AZURE_SPEECH_KEY_ENV = "AZURE_SPEECH_KEY"
AZURE_SPEECH_REGION_ENV = "AZURE_SPEECH_REGION"
AZURE_SPEECH_LANGUAGE_ENV = "AZURE_SPEECH_LANGUAGE"

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
    """Use Azure Speech SDK to choose a target object by voice."""

    def __init__(self, yolo_class_names, default_target="cell phone"):
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

        self.speech_key = os.getenv(AZURE_SPEECH_KEY_ENV)
        self.speech_region = os.getenv(AZURE_SPEECH_REGION_ENV)
        self.speech_language = os.getenv(AZURE_SPEECH_LANGUAGE_ENV, "en-US")

        if not self.speech_key or not self.speech_region:
            raise RuntimeError(
                "Azure Speech SDK is selected, but environment variables "
                f"{AZURE_SPEECH_KEY_ENV} and {AZURE_SPEECH_REGION_ENV} are not set."
            )

        print("Loading Azure Speech SDK...")
        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region,
        )
        self.speech_config.speech_recognition_language = self.speech_language
        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
            "5000",
        )
        self.speech_config.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,
            "800",
        )

        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        self.speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config,
        )
        self.phrase_list = speechsdk.PhraseListGrammar.from_recognizer(self.speech_recognizer)
        self.configure_phrase_list()
        print(f"Azure Speech SDK ready ({self.speech_language})")

    def configure_phrase_list(self):
        """Bias recognition toward supported object names and aliases."""
        for class_name in self.yolo_class_names:
            self.phrase_list.addPhrase(class_name)

        for alias_name in ALIAS_MAP:
            self.phrase_list.addPhrase(alias_name)

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
        """Text-to-speech for short prompts."""
        print(f"[VOICE] {text}")
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    def listen_once(self, prompt_text):
        """Prompt the user and capture one utterance from the default microphone."""
        self.speak(prompt_text)
        print("Listening...")

        result = self.speech_recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = result.text.strip().lower()
            if text:
                print(f"You said: {text}")
                return text
            print("Could not hear clearly.")
            return None

        if result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be matched.")
            return None

        if result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            error_message = details.error_details or details.reason
            print(f"Recognition canceled: {error_message}")
            return None

        print("Recognition did not return usable speech.")
        return None

    def extract_object_from_speech(self, speech_text):
        """Map recognized speech to the nearest supported object name."""
        words = self.normalize_text(speech_text)

        ranked_candidates = self.rank_candidate_objects(words)
        if ranked_candidates:
            return ranked_candidates[0][0]

        for alias in sorted(ALIAS_MAP.keys(), key=len, reverse=True):
            if alias in words:
                return ALIAS_MAP[alias]

        for class_name in sorted(self.yolo_class_names, key=len, reverse=True):
            if class_name in words:
                return class_name

        for word in words.split():
            if word in self.yolo_class_names:
                return word

        return words

    def get_target_object(self):
        """Ask for a target object and return the best supported match."""
        print("=" * 50)
        print("Tell me what object you want to find")
        print("Say only the object name, for example: cup, mouse, bottle, cell phone")
        print("=" * 50)

        for _ in range(self.max_retries):
            speech_text = self.listen_once("Please say only the object name now.")
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
