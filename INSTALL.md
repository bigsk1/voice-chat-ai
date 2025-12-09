# Installation Guide - Voice Chat AI

## ⚠️ BREAKING CHANGE (Python 3.11+ Required)

**XTTS has been removed** and replaced with **Spark-TTS** as the local voice cloning option.

### If you're upgrading from an older version:

1. **Delete your old virtual environment** (`.venv`, `venv`, or conda environment)
2. **Install Python 3.11 or 3.12** if you're still on Python 3.10
3. **Follow one of the installation methods below** to create a fresh environment

### Need to stay on the old XTTS version (Python 3.10)?

If you cannot upgrade, use the last XTTS-compatible commit:

```bash
git checkout d71540f
git checkout -b legacy-xtts
```

This will keep you on Python 3.10 with XTTS, but you won't receive new features or updates.

### What's Changed:

- ❌ **Removed**: Coqui XTTS (slow, outdated)
- ✅ **Added**: Spark-TTS (faster, better quality, optional)
- ✅ **Still Available**: OpenAI TTS, ElevenLabs, Kokoro TTS (all work without Spark-TTS)

---

## Prerequisites

### Linux (Ubuntu/Debian/WSL2)

Install system audio libraries before proceeding:

```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-dev build-essential libasound2-dev libjack-dev ffmpeg
```

### Windows

- Python 3.11 or 3.12 installed
- FFmpeg in PATH ([download](https://ffmpeg.org/download.html))

### macOS

```bash
brew install portaudio ffmpeg
```

---

## Quick Start

### Option 1: Using pip (Recommended)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 2. Install PyTorch (choose ONE based on your hardware)
# CPU only:
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu

# OR CUDA 12.1 (most compatible):
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121

# OR CUDA 12.4 (newest):
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install core dependencies
pip install -r requirements.txt

# 4. (Optional) Install Spark-TTS for local voice cloning
python setup_sparktts.py
```

### Option 2: Using uv (Faster!)

#### Modern approach with pyproject.toml (Recommended)

```bash
# 1. Core app only (no Spark-TTS)
uv sync

# OR with Spark-TTS (install PyTorch first, then sync with extra)

# For CPU:
uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
uv sync --extra sparktts

# For CUDA 12.4:
uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124
uv sync --extra sparktts

# 2. Activate environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

#### Alternative: Manual installation

```bash
# 1. Create virtual environment
uv venv .venv --python 3.11

# 2. Install PyTorch (choose ONE)
# CPU:
uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu --python .venv

# OR CUDA 12.4:
uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124 --python .venv

# 3. Install core dependencies
uv pip install -r requirements.txt --python .venv

# 4. (Optional) Install Spark-TTS
python setup_sparktts.py
```

### Option 3: Using conda

```bash
# 1. Create conda environment
conda create --name voice-chat-ai python=3.11
conda activate voice-chat-ai

# 2. Install PyTorch (choose ONE)
# CPU:
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu

# OR CUDA:
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install core dependencies
pip install -r requirements.txt

# 4. (Optional) Install Spark-TTS
python setup_sparktts.py
```

## What Gets Installed

### Core (requirements.txt)

- FastAPI web framework
- Faster-Whisper (local speech-to-text)
- OpenAI, Anthropic, xAI clients
- Audio processing (PyAudio, librosa, soundfile)
- WebRTC support (for OpenAI Realtime)

### Optional (Spark-TTS)

- Zero-shot voice cloning using character .wav files
- ~5GB model download
- Works with CPU or CUDA GPU

## TTS Provider Options

You can use ANY of these without Spark-TTS:

- ✅ **OpenAI TTS** - Cloud-based, fast, high quality
- ✅ **ElevenLabs** - Cloud-based, excellent quality
- ✅ **Kokoro TTS** - Self-hosted, CPU-friendly, very fast

Add Spark-TTS only if you want **local voice cloning**.

## Python Version

- **Python 3.11+** - Recommended (full PyTorch 2.6+ CUDA support)

## Troubleshooting

### uv installation too slow

Use `--link-mode=copy` to suppress warnings:

```bash
uv pip install -r requirements.txt --python .venv --link-mode=copy
```

## Usage

Run the application: 🏃

Web UI

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Find on http://localhost:8000/

### numpy version conflicts

The requirements.txt pins numpy to `>=1.21.6,<1.28.0` for scipy compatibility.

### PyAudio build errors (Windows)

Install Microsoft C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/

### CUDA not detected

Verify PyTorch installation:

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

If False, reinstall PyTorch with correct CUDA index URL.