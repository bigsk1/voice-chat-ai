# Use the official NVIDIA CUDA base image
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Create a working directory
WORKDIR /app

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    portaudio19-dev \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    wget \
    curl \
    python3.11 \
    python3.11-venv \
    python3-pip \
    alsa-utils \
    alsa-oss \
    alsa-tools \
    pulseaudio \
    libsdl2-dev \
    dbus \
    && rm -rf /var/lib/apt/lists/*

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Install cuDNN
RUN apt-get update && apt-get install -y libcudnn8 libcudnn8-dev

# Configure dbus
RUN dbus-uuidgen > /etc/machine-id

# Configure ALSA to use PulseAudio
RUN echo "pcm.!default pulse" > /root/.asoundrc && \
    echo "ctl.!default pulse" >> /root/.asoundrc

# Ensure the directory exists before writing to the file
RUN mkdir -p /usr/share/alsa/alsa.conf.d && \
    echo "defaults.pcm.card 0" >> /usr/share/alsa/alsa.conf.d/99-pulseaudio-defaults.conf && \
    echo "defaults.ctl.card 0" >> /usr/share/alsa/alsa.conf.d/99-pulseaudio-defaults.conf

# Copy only necessary files (base app without Spark-TTS models)
COPY requirements.txt /app/requirements.txt
COPY app /app/app
COPY characters /app/characters
COPY outputs /app/outputs
COPY cli.py /app/cli.py
COPY elevenlabs_voices.json.example /app/elevenlabs_voices.json.example
COPY .env.sample /app/.env.sample
COPY setup_sparktts.py /app/setup_sparktts.py
COPY download_sparktts_model.py /app/download_sparktts_model.py
COPY requirements_sparktts.txt /app/requirements_sparktts.txt
COPY cli /app/cli
COPY sparktts /app/sparktts

# Install Python dependencies (without cache to reduce image size)
RUN pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

# Note: Spark-TTS is NOT pre-installed to keep image lean
# Users can optionally run: python setup_sparktts.py inside the container
# This will install PyTorch with CUDA support + Spark-TTS dependencies

# Expose the port that the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
