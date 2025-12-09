# Spark-TTS - Optional Local Voice Cloning

Spark-TTS is an **optional** LLM-based TTS model (0.5B parameters) that provides zero-shot voice cloning using your character `.wav` files.

## Why Spark-TTS?

- ✅ **Zero-shot voice cloning** - uses character voice files
- ✅ **Fully local** - no API keys needed
- ✅ **GPU accelerated** - fast inference with CUDA
- ✅ **Cross-platform** - Windows, Linux, macOS

## Requirements

- **Python**: 3.11+ (recommended for CUDA support)
- **GPU**: CUDA-capable GPU (optional, will use CPU)
- **Disk**: ~5GB for model files
- **RAM**: 8GB+ (16GB+ recommended)

## Quick Setup

### Automated Setup (Recommended)

```bash
# GPU (CUDA) version
python setup_sparktts.py

# CPU-only version
python setup_sparktts.py --cpu-only
```

### Manual Setup

1. **Install dependencies:**

   ```bash
   # For GPU (CUDA 12.4)
   pip install -r requirements_cuda_sparktts.txt
   
   # For CPU only
   pip install -r requirements_cpu_sparktts.txt
   ```

2. **Download model (~4GB):**

   ```bash
   python download_sparktts_model.py
   ```

3. **Configure `.env`:**

   ```bash
   TTS_PROVIDER=sparktts
   SPARKTTS_MODEL_DIR=pretrained_models/Spark-TTS-0.5B
   SPARKTTS_MAX_CHARS=1000
   ```

4. **Run the app:**

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## Python Version Notes

- **Python 3.11+**: Recommended - full PyTorch 2.6+ CUDA support

## Performance

| Hardware | Speed | Quality |
|----------|-------|---------|
| CPU | Slow (~30-60s) | Good |
| GPU (CUDA) | Fast (~2-5s) | Good |

## Troubleshooting

### "TTS Provider is blank in UI"

- Ensure `TTS_PROVIDER=sparktts` in `.env`
- Restart the application

### "CUDA not available"

```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall CUDA PyTorch
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124
```

### "numpy version incompatible"

```bash
pip install "numpy>=1.21.6,<1.28.0"
```

### Model download fails

- Check internet connection
- Ensure ~5GB free disk space
- Try manual download from [Hugging Face](https://huggingface.co/bigsk1/Spark-TTS-0.5B)

## Alternative TTS Options

Spark-TTS is **optional**. The app also supports:

- **OpenAI TTS** - Cloud-based, fast, high quality
- **ElevenLabs** - Cloud-based, excellent quality
- **Kokoro TTS** - Local, CPU-friendly, very fast

## Files Overview

```
voice-chat-ai/
├── sparktts/              # Spark-TTS core modules (integrated)
├── cli/                   # SparkTTS class and inference
├── download_sparktts_model.py  # Model download script
├── setup_sparktts.py      # Automated setup script
├── requirements_cpu_sparktts.txt   # CPU dependencies
├── requirements_cuda_sparktts.txt  # CUDA dependencies
└── pretrained_models/     # Model storage (gitignored)
```

## Uninstalling

To remove Spark-TTS:

1. Delete `pretrained_models/` folder
2. Delete `sparktts/` and `cli/` folders
3. Set `TTS_PROVIDER` to another option in `.env`
4. Remove Spark-TTS packages: `pip uninstall einx einops omegaconf hydra-core soxr`

---

**Note**: This is an experimental feature. For production use, consider OpenAI or ElevenLabs TTS.