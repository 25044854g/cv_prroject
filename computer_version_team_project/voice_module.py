"""
Voice Recognition Module
Use Whisper for speech-to-text and OpenRouter for semantic mapping
from natural-language requests to YOLO object labels.
"""

import json
import os
import re
import tempfile
import wave
from difflib import SequenceMatcher
from urllib import error, request

import pyaudio
import pyttsx3
import whisper


OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_MODEL_ENV = "OPENROUTER_MODEL"
OPENROUTER_SITE_URL_ENV = "OPENROUTER_SITE_URL"
OPENROUTER_APP_NAME_ENV = "OPENROUTER_APP_NAME"
WHISPER_MODEL_ENV = "WHISPER_MODEL"
WHISPER_LANGUAGE_ENV = "WHISPER_LANGUAGE"


def load_local_env(env_path):
    """Load simple KEY=VALUE pairs from a local .env file if present."""
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue

            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


load_local_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Ensure ffmpeg is available for Whisper through imageio-ffmpeg when installed.
try:
    import imageio_ffmpeg

    _ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    _ffmpeg_dir = os.path.dirname(_ffmpeg_exe)
    _link_path = os.path.join(_ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(_link_path):
        import shutil

        shutil.copy2(_ffmpeg_exe, _link_path)
    if _ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except ImportError:
    pass


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

YES_WORDS = {
    "yes",
    "yeah",
    "yep",
    "correct",
    "right",
    "ok",
    "okay",
    "sure",
    "affirmative",
}

NO_WORDS = {
    "no",
    "nope",
    "wrong",
    "incorrect",
    "not",
    "negative",
}


class VoiceDetector:
    """Use Whisper and OpenRouter to choose a target object by voice."""

    def __init__(self, yolo_class_names, default_target="cell phone"):
        self.yolo_class_names = [name.lower() for name in yolo_class_names]
        self.default_target = default_target
        self.max_retries = 5
        self.max_confirmation_retries = 3
        self.min_match_score = 0.55
        self.candidate_display_count = 3

        self.normalized_class_map = {
            self.normalize_text(name): name for name in self.yolo_class_names
        }
        self.normalized_alias_map = {
            self.normalize_text(alias): target for alias, target in ALIAS_MAP.items()
        }

        self.openrouter_api_key = os.getenv(OPENROUTER_API_KEY_ENV)
        self.openrouter_model = os.getenv(OPENROUTER_MODEL_ENV, "openai/gpt-4o-mini")
        self.openrouter_site_url = os.getenv(OPENROUTER_SITE_URL_ENV, "http://localhost")
        self.openrouter_app_name = os.getenv(OPENROUTER_APP_NAME_ENV, "computer-version-team-project")
        self.whisper_model_name = os.getenv(WHISPER_MODEL_ENV, "base")
        self.whisper_language = os.getenv(WHISPER_LANGUAGE_ENV, "zh")

        if not self.openrouter_api_key:
            raise RuntimeError(
                "Missing OpenRouter configuration. Create a .env file in the project root "
                f"and set {OPENROUTER_API_KEY_ENV}. See .env.example for the expected format."
            )

        print(f"Loading Whisper model ({self.whisper_model_name})...")
        self.whisper_model = whisper.load_model(self.whisper_model_name)
        print(f"Whisper ready ({self.whisper_language})")
        print(f"OpenRouter model ready ({self.openrouter_model})")

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

    def call_openrouter(self, user_request):
        """Ask OpenRouter to map a natural-language request to one YOLO label."""
        payload = {
            "model": self.openrouter_model,
            "temperature": 0,
            "max_tokens": 150,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You map a spoken request to exactly one object label from a fixed candidate list. "
                        "Return JSON only in the form {\"label\": \"...\", \"reason\": \"...\"}. "
                        "The label must be one of the candidate labels exactly, or NONE if nothing matches."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Candidate labels: {json.dumps(self.yolo_class_names, ensure_ascii=False)}\n"
                        f"Helpful aliases: {json.dumps(ALIAS_MAP, ensure_ascii=False)}\n"
                        f"User request: {user_request}"
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
        }

        req = request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.openrouter_site_url,
                "X-Title": self.openrouter_app_name,
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=30) as response:
                response_json = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {error_body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"OpenRouter connection error: {exc.reason}") from exc

        try:
            content = response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected OpenRouter response: {response_json}") from exc

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"OpenRouter did not return valid JSON: {content}") from exc

        label = parsed.get("label")
        reason = parsed.get("reason", "")
        if not isinstance(label, str):
            raise RuntimeError(f"OpenRouter response missing label: {parsed}")

        return label.strip(), str(reason).strip()

    def match_object_with_llm(self, speech_text):
        """Use OpenRouter first, then fall back to local fuzzy ranking if needed."""
        try:
            label, reason = self.call_openrouter(speech_text)
            normalized_label = self.normalize_text(label)

            if normalized_label == "none":
                return None, reason or "The model could not map the request to a supported label."

            for class_name in self.yolo_class_names:
                if normalized_label == self.normalize_text(class_name):
                    return class_name, reason

            ranked_candidates = self.rank_candidate_objects(label)
            if ranked_candidates:
                return ranked_candidates[0][0], reason or f"Model returned {label}."

            return None, reason or f"Model returned unsupported label: {label}"
        except RuntimeError as exc:
            print(f"[VOICE] OpenRouter API failed, falling back to local matching: {exc}")
            ranked_candidates = self.rank_candidate_objects(speech_text)
            if ranked_candidates:
                return ranked_candidates[0][0], f"Local fuzzy match (API failed: {exc})"
            return None, f"Fallback match failed: {exc}"

    def speak(self, text):
        """Text-to-speech for short prompts."""
        print(f"[VOICE] {text}")
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    def record_audio(self, duration=5):
        """Record audio from the default microphone to a temporary wav file."""
        audio_interface = pyaudio.PyAudio()
        stream = audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024,
        )

        print(f"Recording for {duration} seconds...")
        frames = []
        total_chunks = int(16000 / 1024 * duration)
        for _ in range(total_chunks):
            frames.append(stream.read(1024, exception_on_overflow=False))

        stream.stop_stream()
        stream.close()
        sample_width = audio_interface.get_sample_size(pyaudio.paInt16)
        audio_interface.terminate()

        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.close()

        with wave.open(temp_file.name, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"".join(frames))

        return temp_file.name

    def listen_once(self, prompt_text):
        """Prompt the user and capture one utterance from the default microphone."""
        self.speak(prompt_text)
        audio_path = None
        try:
            audio_path = self.record_audio(duration=5)
            result = self.whisper_model.transcribe(
                audio_path,
                language=self.whisper_language,
                fp16=False,
            )
            text = result.get("text", "").strip().lower()
            if text:
                print(f"You said: {text}")
                return text
            print("Could not hear clearly.")
            return None
        except Exception as exc:
            print(f"Recognition error: {exc}")
            return None
        finally:
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)

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

    def is_yes_response(self, speech_text):
        """Return True when the confirmation response is affirmative."""
        normalized = self.normalize_text(speech_text)
        if not normalized:
            return False
        return any(word in normalized.split() or word in normalized for word in YES_WORDS)

    def is_no_response(self, speech_text):
        """Return True when the confirmation response is negative."""
        normalized = self.normalize_text(speech_text)
        if not normalized:
            return False
        return any(word in normalized.split() or word in normalized for word in NO_WORDS)

    def get_target_object(self):
        """Ask for a target object and return the best supported match."""
        print("=" * 50)
        print("Tell me what object you want to find")
        print("You can say a full sentence, for example: find my phone on the desk")
        print("=" * 50)

        for _ in range(self.max_retries):
            speech_text = self.listen_once("Please say what object you want to find.")
            if speech_text is None:
                continue

            extracted, reason = self.match_object_with_llm(speech_text)

            if extracted:
                print(f"Matched target: {extracted}")
                if reason:
                    print(f"Match reason: {reason}")

                confirmation_accepted = False
                user_rejected = False

                for confirmation_attempt in range(self.max_confirmation_retries):
                    if confirmation_attempt == 0:
                        prompt = f"I heard {extracted}. Is that correct? Please say yes or no."
                    else:
                        prompt = "I still need a yes or no answer. Please say yes or no."

                    confirmation = self.listen_once(prompt)

                    if confirmation is None:
                        print("Confirmation not heard clearly.")
                        if confirmation_attempt < self.max_confirmation_retries - 1:
                            self.speak("I did not hear your answer clearly. Please say yes or no.")
                        continue

                    if self.is_yes_response(confirmation):
                        confirmation_accepted = True
                        break

                    if self.is_no_response(confirmation):
                        user_rejected = True
                        break

                    print(f"Unclear confirmation response: {confirmation}")
                    if confirmation_attempt < self.max_confirmation_retries - 1:
                        self.speak("Please answer with yes or no.")

                if confirmation_accepted:
                    self.speak(f"OK, looking for {extracted}")
                    print(f"Confirmed target: {extracted}\n")
                    return extracted

                if user_rejected:
                    print("User rejected matched target. Asking again.\n")
                    self.speak("Okay, please say the object again.")
                    continue

                print("Confirmation failed after multiple attempts. Asking for the object again.\n")
                self.speak("I could not confirm the object. Please say the object again.")
                continue

            print("The request did not match a supported object.")
            if reason:
                print(f"Match reason: {reason}")
            self.speak("I could not map that request to a supported object. Please try again.")

        print(f"Too many retries. Using default target: {self.default_target}")
        self.speak(f"Using default target: {self.default_target}")
        return self.default_target
