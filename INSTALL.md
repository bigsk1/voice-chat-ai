# Installation Guide - Voice Chat AI

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
- **Python 3.10** - Works but limited to PyTorch 2.5.x (no Spark-TTS CUDA)

## Troubleshooting

### uv installation too slow
Use `--link-mode=copy` to suppress warnings:
```bash
uv pip install -r requirements.txt --python .venv --link-mode=copy
```

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

