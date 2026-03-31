# Hand-Object Detection with Voice Control

A computer vision project that uses hand detection and object detection to help users locate and grab objects through voice commands.

## Features

- **Hand Detection**: Real-time hand tracking using MediaPipe
- **Object Detection**: YOLO-based object detection
- **Depth Detection**: Estimates hand-object depth relationship
- **Voice Control**: Voice commands to specify target objects
- **Audio Guidance**: Voice instructions for hand positioning
- **Real-time Feedback**: Visual and audio feedback during interaction

## Requirements

- Python 3.8+
- OpenCV (cv2)
- MediaPipe
- YOLOv8
- PyAudio
- pyttsx3
- Whisper (OpenAI)
- OpenRouter API (for LLM-based object matching)

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root with your OpenRouter API configuration:

```
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_SITE_URL=http://localhost
OPENROUTER_APP_NAME=computer-version-team-project
WHISPER_MODEL=base
WHISPER_LANGUAGE=zh
```

## Usage

```bash
python main.py
```

The application will:
1. Ask for a target object via voice
2. Open the camera and listen for hand/object position
3. Provide voice guidance for hand positioning
4. Display "GET IT!" when the hand is close enough to grab

Press 'q' to exit.

## Project Structure

- `main.py` - Main application entry point
- `hand_detection_module.py` - Hand and object detection
- `depth_detection_module.py` - Depth analysis
- `voice_module.py` - Voice recognition and synthesis

## How It Works

1. **Voice Input**: User says what object to find
2. **Detection Phase**: System detects hand and target object
3. **Alignment Phase**: Provides guidance to align hand with object
4. **Grab Phase**: When hand is close enough, says "GET IT!"

## Controls

- Press 'q' to exit the application
- Voice commands: Say the object name you want to find

## License

MIT License
