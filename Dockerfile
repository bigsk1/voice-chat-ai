# Use the official NVIDIA CUDA base image - This total image is huge 40 gb when completly built - use at your own risk!
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Create a working directory
WORKDIR /app

# Install system dependencies and Python 3.10
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
    python3.10 \
    python3.10-venv \
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

# Copy the requirements file
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy the rest of the application code
COPY . /app

# Set the working directory
WORKDIR /app

# Expose the port that the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
