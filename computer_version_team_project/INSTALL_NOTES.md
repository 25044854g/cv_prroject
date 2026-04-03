# Installation Notes

## System Dependencies

### Ubuntu/Debian
Before running `pip install -r requirements.txt`, install required system libraries:

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev
```

### macOS
```bash
brew install portaudio
```

### Windows
PyAudio pre-built wheels should work. If not, install from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

## Python Dependencies
After installing system dependencies, run:

```bash
pip install -r requirements.txt
```

## Troubleshooting

**Issue: `portaudio.h: No such file or directory`**
- Solution: Install portaudio19-dev (Ubuntu) or portaudio (macOS)

**Issue: PyAudio wheel not found**
- Solution: Try pre-built wheels or use a Python version <= 3.11
