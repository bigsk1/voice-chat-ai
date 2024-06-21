#!/bin/bash

# Check if two arguments are passed
if [ $# -ne 2 ]; then
    echo "Usage: $0 <MODEL_PROVIDER> <TTS_PROVIDER>"
    exit 1
fi

MODEL_PROVIDER=$1
TTS_PROVIDER=$2

# Update the .env file with the new model provider and TTS provider
sed -i "s/^MODEL_PROVIDER=.*/MODEL_PROVIDER=$MODEL_PROVIDER/" .env
sed -i "s/^TTS_PROVIDER=.*/TTS_PROVIDER=$TTS_PROVIDER/" .env

# Restart the Docker container
docker restart voice-chat-ai


# docker run -d --gpus all -e "PULSE_SERVER=/mnt/wslg/PulseServer" -v /mnt/wslg/:/mnt/wslg/ --env-file docker/.env --name voice-chat-ai -p 8000:8000 voice-chat-ai:latest
# make sure to change permissions after download  chmod +x update_providers.sh
# run to update as needed  ./update_providers.sh ollama xtts
