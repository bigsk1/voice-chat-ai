#!/usr/bin/env python3
"""
Download Spark-TTS pretrained model from Hugging Face.
This script downloads the Spark-TTS-0.5B model to the pretrained_models directory.
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

def download_model():
    """Download Spark-TTS model from Hugging Face."""
    model_dir = Path("pretrained_models/Spark-TTS-0.5B")
    
    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"Model files already exist in {model_dir}. Skipping download.")
        print("If you want to re-download, delete the directory first.")
        return
    
    print("Downloading Spark-TTS-0.5B model from Hugging Face...")
    print("This may take several minutes depending on your internet connection.")
    
    try:
        # Using forked model for stability (Apache 2.0 licensed)
        snapshot_download(
            repo_id="bigsk1/Spark-TTS-0.5B",
            local_dir=str(model_dir),
            resume_download=True
        )
        print(f"\n✓ Model downloaded successfully to {model_dir}")
        print("You can now use Spark-TTS in your application.")
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("Please check your internet connection and try again.")
        raise

if __name__ == "__main__":
    download_model()

