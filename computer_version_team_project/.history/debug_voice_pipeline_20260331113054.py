"""Debug script for the voice pipeline without touching the main app flow."""

from datetime import datetime
import os

from ultralytics import YOLO

from voice_module import VoiceDetector


LOG_FILE_NAME = "debug_voice_pipeline.log"


def load_yolo_class_names():
    """Load class names from the local YOLO model used by the project."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "yolov8m.pt")
    model = YOLO(model_path)
    return list(model.names.values())


def append_debug_log(transcript, local_candidates, matched_label=None, reason=None, error_message=None):
    """Append one debug attempt to a local log file."""
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILE_NAME)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    candidate_text = ", ".join(
        f"{label} ({score:.0%})" for label, score in local_candidates
    ) if local_candidates else "none"

    lines = [
        f"[{timestamp}]",
        f"transcript: {transcript}",
        f"local_candidates: {candidate_text}",
    ]

    if matched_label is not None:
        lines.append(f"matched_label: {matched_label}")
    if reason:
        lines.append(f"reason: {reason}")
    if error_message:
        lines.append(f"error: {error_message}")

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write("\n".join(lines) + "\n\n")


def main():
    yolo_class_names = load_yolo_class_names()
    detector = VoiceDetector(yolo_class_names)
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILE_NAME)

    print("=" * 60)
    print("Voice Pipeline Debug")
    print("This script only tests Whisper transcription and OpenRouter label matching.")
    print(f"Log file: {log_path}")
    print("Press Ctrl+C to stop.")
    print("=" * 60)

    while True:
        try:
            transcript = detector.listen_once("Please describe the object you want to find.")
            if transcript is None:
                print("Transcription failed. Try again.\n")
                append_debug_log("<no transcript>", [], error_message="Transcription failed")
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
                append_debug_log(transcript, local_candidates, error_message=str(exc))
                continue

            print(f"Matched label: {matched_label}")
            print(f"Reason: {reason or 'No reason returned'}")
            print()
            append_debug_log(transcript, local_candidates, matched_label=matched_label, reason=reason)
        except KeyboardInterrupt:
            print("\nStopped voice debug.")
            break


if __name__ == "__main__":
    main()