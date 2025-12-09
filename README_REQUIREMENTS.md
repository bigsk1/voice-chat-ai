# Requirements Files Guide

## For New Installations (Python 3.11+)

### Standard Installation
Use these for most users:

1. **`requirements.txt`** - Core dependencies (no TTS models included)
   - Works with OpenAI, ElevenLabs, Kokoro TTS
   - Install PyTorch separately (see INSTALL.md)
   
2. **`requirements_sparktts.txt`** - Optional Spark-TTS add-on
   - Install AFTER requirements.txt
   - Only needed for local voice cloning

### All-in-One Installation
For users who want everything in one file:

- **`requirements_cpu_sparktts.txt`** - CPU version with Spark-TTS
- **`requirements_cuda_sparktts.txt`** - CUDA version with Spark-TTS

## For Legacy Python 3.10

- **`requirements_cpu_legacy_py310.txt`** - Old CPU requirements with XTTS
  - Use only if you must stay on Python 3.10
  - Not recommended for new installations

## Installation Methods

### Method 1: Modular (Recommended)
```bash
# Install PyTorch first
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124

# Install core
pip install -r requirements.txt

# Optional: Add Spark-TTS
pip install -r requirements_sparktts.txt
python download_sparktts_model.py
```

### Method 2: All-in-One
```bash
# Everything including Spark-TTS
uv pip install -r requirements_cuda_sparktts.txt --python .venv --index-strategy unsafe-best-match
python download_sparktts_model.py
```

### Method 3: Automated
```bash
# One command setup
python setup_sparktts.py
```

## Which File Should I Use?

| Scenario | File to Use |
|----------|-------------|
| New user, any TTS provider | `requirements.txt` |
| Want local voice cloning | `requirements.txt` + `requirements_sparktts.txt` |
| Simple all-in-one install | `requirements_cuda_sparktts.txt` or `requirements_cpu_sparktts.txt` |
| Stuck on Python 3.10 | `requirements_cpu_legacy_py310.txt` |

## Notes

- All files work with **pip**, **uv**, or **conda**
- PyTorch should be installed separately for best control
- Spark-TTS is **optional** - the app works great without it

