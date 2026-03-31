"""Debug script for the voice pipeline without touching the main app flow."""

import os

from ultralytics import YOLO

from voice_module import VoiceDetector


def load_yolo_class_names():
    """Load class names from the local YOLO model used by the project."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "yolov8m.pt")
    model = YOLO(model_path)
    return list(model.names.values())


def main():
    yolo_class_names = load_yolo_class_names()
    detector = VoiceDetector(yolo_class_names)

    print("=" * 60)
    print("Voice Pipeline Debug")
    print("This script only tests Whisper transcription and OpenRouter label matching.")
    print("Press Ctrl+C to stop.")
    print("=" * 60)

    while True:
        try:
            transcript = detector.listen_once("Please describe the object you want to find.")
            if transcript is None:
                print("Transcription failed. Try again.\n")
                continue

            print(f"Whisper transcript: {transcript}")

            local_candidates = detector.rank_candidate_objects(transcript)
            if local_candidates:
                preview = ", ".join(
                    f"{label} ({score:.0%})" for label, score in local_candidates
                )
                print(f"Local fuzzy candidates: {preview}")
            else:
                print("Local fuzzy candidates: none")

            try:
                matched_label, reason = detector.match_object_with_llm(transcript)
            except RuntimeError as exc:
                print(f"OpenRouter matching error: {exc}\n")
                continue

            print(f"Matched label: {matched_label}")
            print(f"Reason: {reason or 'No reason returned'}")
            print()
        except KeyboardInterrupt:
            print("\nStopped voice debug.")
            break


if __name__ == "__main__":
    main()