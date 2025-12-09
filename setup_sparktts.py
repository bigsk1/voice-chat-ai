#!/usr/bin/env python3
"""
Optional Spark-TTS Setup Script
================================
This script automates the setup of Spark-TTS for local voice cloning.
Spark-TTS is OPTIONAL - the app works with OpenAI, ElevenLabs, and Kokoro TTS without it.

Requirements:
- Python 3.11+ (for best CUDA PyTorch support)
- CUDA-capable GPU (optional, will use CPU otherwise)
- ~5GB free disk space for model

Usage:
    python setup_sparktts.py [--cpu-only]
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.11+"""
    if sys.version_info < (3, 11):
        print(f"⚠️  Python 3.11+ recommended (you have {sys.version_info.major}.{sys.version_info.minor})")
        print("   Spark-TTS will work but PyTorch 2.6+ CUDA support requires Python 3.11+")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

def check_uv_venv():
    """Check if we're in a uv-created venv (no pip by default)"""
    # Check if uv is available and pip is not
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, check=True)
            return False  # pip exists, not a uv-only venv
        except subprocess.CalledProcessError:
            return True  # uv exists but pip doesn't
    except:
        return False

def install_dependencies(cpu_only=False):
    """Install Spark-TTS dependencies"""
    
    print(f"\n📦 Installing dependencies...")
    print("   This will take 5-10 minutes...")
    
    # Detect if we should use uv pip or pip
    use_uv = check_uv_venv()
    
    if use_uv:
        print("   Detected uv environment, using uv pip...")
        pip_cmd = ["uv", "pip", "install"]
        pip_args = ["--python", sys.executable]
    else:
        pip_cmd = [sys.executable, "-m", "pip", "install"]
        pip_args = []
    
    try:
        # Step 1: Install PyTorch
        if cpu_only:
            print("   Installing CPU PyTorch...")
            subprocess.run(
                pip_cmd + ["torch", "torchaudio", "torchvision", 
                          "--index-url", "https://download.pytorch.org/whl/cpu"] + pip_args,
                check=True
            )
        else:
            print("   Installing CUDA PyTorch (12.4)...")
            subprocess.run(
                pip_cmd + ["torch", "torchaudio", "torchvision",
                          "--index-url", "https://download.pytorch.org/whl/cu124"] + pip_args,
                check=True
            )
        
        # Step 2: Install core dependencies
        print("   Installing core dependencies...")
        subprocess.run(
            pip_cmd + ["-r", "requirements.txt"] + pip_args,
            check=True
        )
        
        # Step 3: Install Spark-TTS specific dependencies
        print("   Installing Spark-TTS dependencies...")
        subprocess.run(
            pip_cmd + ["-r", "requirements_sparktts.txt"] + pip_args,
            check=True
        )
        
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def download_model():
    """Download Spark-TTS model"""
    model_dir = Path("pretrained_models/Spark-TTS-0.5B")
    
    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"\n✅ Model already exists in {model_dir}")
        response = input("   Re-download? (y/n): ")
        if response.lower() != 'y':
            return True
    
    print("\n📥 Downloading Spark-TTS model (~4GB)...")
    print("   This will take several minutes...")
    
    try:
        from huggingface_hub import snapshot_download
        
        snapshot_download(
            repo_id="SparkAudio/Spark-TTS-0.5B",
            local_dir=str(model_dir),
            resume_download=True
        )
        print(f"✅ Model downloaded to {model_dir}")
        return True
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False

def update_env_file():
    """Update or create .env file with Spark-TTS settings"""
    env_file = Path(".env")
    env_sample = Path(".env.sample")
    
    # Read existing .env or create from sample
    if env_file.exists():
        with open(env_file, 'r') as f:
            lines = f.readlines()
    elif env_sample.exists():
        with open(env_sample, 'r') as f:
            lines = f.readlines()
    else:
        lines = []
    
    # Check if TTS_PROVIDER is already set
    has_tts_provider = any(line.startswith('TTS_PROVIDER=') for line in lines)
    has_sparktts_dir = any(line.startswith('SPARKTTS_MODEL_DIR=') for line in lines)
    
    if not has_tts_provider or not has_sparktts_dir:
        print("\n⚙️  Updating .env file...")
        
        if not has_tts_provider:
            lines.append("\n# Spark-TTS Configuration\n")
            lines.append("# TTS_PROVIDER=sparktts\n")
        
        if not has_sparktts_dir:
            lines.append("SPARKTTS_MODEL_DIR=pretrained_models/Spark-TTS-0.5B\n")
            lines.append("SPARKTTS_MAX_CHARS=1000\n")
        
        with open(env_file, 'w') as f:
            f.writelines(lines)
        
        print(f"✅ Updated {env_file}")
        print("   Note: TTS_PROVIDER is commented out - uncomment to use Spark-TTS by default")
    else:
        print(f"\n✅ {env_file} already configured for Spark-TTS")

def main():
    """Main setup function"""
    print("=" * 60)
    print("   Spark-TTS Setup - Local Voice Cloning")
    print("=" * 60)
    
    # Parse arguments
    cpu_only = "--cpu-only" in sys.argv
    
    if cpu_only:
        print("\n🖥️  CPU-only mode selected")
    else:
        print("\n🚀 GPU (CUDA) mode selected")
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    if not install_dependencies(cpu_only):
        print("\n❌ Setup failed at dependency installation")
        sys.exit(1)
    
    # Download model
    if not download_model():
        print("\n❌ Setup failed at model download")
        sys.exit(1)
    
    # Update .env
    update_env_file()
    
    # Success message
    print("\n" + "=" * 60)
    print("✅ Spark-TTS setup complete!")
    print("=" * 60)
    print("\nTo use Spark-TTS:")
    print("  1. Uncomment 'TTS_PROVIDER=sparktts' in .env file")
    print("  2. Run: uvicorn app.main:app --host 0.0.0.0 --port 8000")
    print("  3. Select 'Spark-TTS (Local)' in the UI")
    print("\nNote: Spark-TTS is optional. You can use OpenAI, ElevenLabs, or Kokoro TTS without it.")
    print("")

if __name__ == "__main__":
    main()

