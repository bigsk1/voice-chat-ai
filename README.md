[![Python application](https://github.com/bigsk1/voice-chat-ai/actions/workflows/python-app.yml/badge.svg)](https://github.com/bigsk1/voice-chat-ai/actions/workflows/python-app.yml)
![Docker support](https://img.shields.io/badge/docker-supported-blue)
[![License](https://img.shields.io/github/license/bigsk1/voice-chat-ai)](https://github.com/bigsk1/voice-chat-ai/blob/main/LICENSE)

# Voice Chat AI 🎙️

Voice Chat AI is a project that allows you to interact with different AI characters using speech. You can choose between various characters, each with unique personalities and voices. Have a serious conversation with Albert Einstein or role play with the OS from the movie HER.

You can run all locally, you can use openai for chat and voice, you can mix between the two. You can use ElevenLabs voices with ollama models all controlled from a Web UI. Ask the AI to look at your screen and it will explain in detail what it's looking at.

New -  WebRTC Real Time API you can have a real time conversation, interrupt the AI and have instant responses. You can also use OpenAI's new TTS model to make the AI more human like with emotions and expressive voices.

## Quick Start

Get up and running fast with Voice Chat AI! 🔊

- [**Install Locally**](#installation): Set up with Python 3.10 on Windows or Linux.
- [**Run with Docker**](#install-with-docker): Use Docker run or Docker Compose
- [**Configure Settings**](#configuration): Customize AI models, voices, and characters via `.env`.
- [**OpenAI Enhanced**](#openai-enhanced): Use OpenAI Enhanced Mode to speak with the AI in a more human like way with emotions.
- [**OpenAI Realtime**](#openai-realtime): Experience real-time conversations with OpenAI's WebRTC-based Realtime API.
- [**Add New Characters**](#adding-new-characters): Add new characters to the project.
- [**Troubleshooting**](#troubleshooting): Fix common audio or CUDA errors.

![Ai-Speech](https://imagedelivery.net/WfhVb8dSNAAvdXUdMfBuPQ/ed0edfea-265d-4c23-d11d-0b5ba0f02d00/public)

## Features

- **Supports OpenAI, xAI or Ollama language models**: Choose the model that best fits your needs.
- **Provides text-to-speech synthesis using XTTS or OpenAI TTS or ElevenLabs**: Enjoy natural and expressive voices.
- **NEW OpenAI Enhanced Mode TTS Model**: Uses emotions and prompts to make the AI more human like.
- **Flexible transcription options**: Uses OpenAI transcription by default, with option to use Local Faster Whisper (automatically downloads when selected).
- **No typing needed, just speak**: Hands-free interaction makes conversations smooth and effortless.
- **Analyzes user mood and adjusts AI responses accordingly**: Get personalized responses based on your mood.
- **You can, just by speaking, have the AI analyze your screen and chat about it**: Seamlessly integrate visual context into your conversations.
- **Easy configuration through environment variables**: Customize the application to suit your preferences with minimal effort.
- **WebUI or Terminal usage**: Run with your preferred method , but recommend the ui as you can change characters, model providers, speech providers, voices, ect..
- **HUGE selection of built in Characters**: Talk with the funniest and most insane AI characters!
- **Docker Support**: Prebuilt image from dockerhub or build yor own image with or without nvidia cuda. Can run on CPU only.

https://github.com/user-attachments/assets/f4401acf-4422-458f-bcbc-06ff63de010e

## Installation

### Requirements

- Python 3.10
- ffmpeg
- Ollama models or Openai API or xAI for chat
- Local XTTS or Openai API or ElevenLabs API for speech
- Microsoft C++ Build Tools on windows
- Microphone
- A sense of humor

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/bigsk1/voice-chat-ai.git
   cd voice-chat-ai
   ```

2. Create a virtual environment: 🐍

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

    On Windows use `venv\Scripts\Activate`

   or use `conda` just make it python 3.10

   ```bash
   conda create --name voice-chat-ai python=3.10
   conda activate voice-chat-ai
   ```

3. Install dependencies:

    Windows Only if using XTTS: Need to have Microsoft C++ 14.0 or greater Build Tools on windows.
    [Microsoft Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

   For GPU (CUDA) version: RECOMMEND

    Install CUDA-enabled PyTorch and other dependencies

    ```bash
   pip install torch==2.3.1+cu121 torchaudio==2.3.1+cu121 torchvision==0.18.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
   ```

   ```bash
   pip install -r requirements.txt
   ```

    To install cpu use only use:

    ```bash
    pip install -r requirements_cpu.txt
    ```

    Make sure you have ffmpeg downloaded, on windows terminal ( winget install ffmpeg ) or checkout https://ffmpeg.org/download.html then restart shell or vscode, type ffmpeg -version to see if installed correctly

    Note: The app uses OpenAI transcription by default. If you select Local Faster Whisper in the UI, it will automatically download the model (about 1GB) on first use. The model is stored in your user's cache directory and shared across environments.

    Local XTTS can run on cpu but is slow, if using a enabled cuda gpu you also might need cuDNN for using nvidia GPU https://developer.nvidia.com/cudnn  and make sure `C:\Program Files\NVIDIA\CUDNN\v9.5\bin\12.6`
is in system PATH or whatever version you downloaded, you can also disable cudnn in the `"C:\Users\Your-Name\AppData\Local\tts\tts_models--multilingual--multi-dataset--xtts_v2\config.json"` to `"cudnn_enable": false`, if you don't want to use it.

### XTTS for local voices - Optional

If you are only using speech with Openai or Elevenlabs then you don't need this. To use the local TTS the first time you select XTTS the model will download and be ready to use, if your device is cuda enabled it will load into cuda if not will fall back to cpu.

## Usage

Run the application: 🏃

Web UI

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Find on http://localhost:8000/

## Terminal Usage

Terminal Usage is also supported, it's a feature rich CLI that allows you to speak with the AI. Update your changes in the .env file rename elevenlabs_voices.json.example to elevenlabs_voices.json and run the cli.py file.

```bash
python3 cli.py
```

## Install with Docker

### 📄 Prerequisites

1. Docker installed on your system.
2. A `.env` file in the same folder as the command. This file should contain all necessary environment variables for the application.

---

### 🐳 Docker compose

uncomment the lines needed in the docker-compose.yml depending on your host system, image pulls latest from dockerhub

```yaml
services:
  voice-chat-ai:
    image: bigsk1/voice-chat-ai:latest
    container_name: voice-chat-ai
    environment:
      - PULSE_SERVER=/mnt/wslg/PulseServer  # Default: WSL2 PulseAudio server (Windows CMD or WSL2 Ubuntu)
      # - PULSE_SERVER=unix:/tmp/pulse/native  # Uncomment for native Ubuntu/Debian with PulseAudio
    env_file:
      - .env
    volumes:
      - \\wsl$\Ubuntu\mnt\wslg:/mnt/wslg/  # Default: WSL2 audio mount for Windows CMD with Docker Desktop
      # - /mnt/wslg/:/mnt/wslg/  # Uncomment for WSL2 Ubuntu (running Docker inside WSL2 distro)
      # - ~/.config/pulse/cookie:/root/.config/pulse/cookie:ro  # Uncomment for native Ubuntu/Debian
      # - /run/user/1000/pulse:/tmp/pulse:ro  # Uncomment and adjust UID (e.g., 1000) for native Ubuntu/Debian
      # - ./elevenlabs_voices.json:/app/elevenlabs_voices.json  # Add your own voice IDs
    ports:
      - "8000:8000"
    restart: unless-stopped
    tty: true  # Enable CLI interactivity (e.g., cli.py)
    stdin_open: true  # Keep STDIN open for interactive use
```

```bash
docker-compose up -d
```

### 🐳 Docker run

### without Nvidia Cuda - cpu mode

Cuda and cudnn not supported. No gpu is used and slower when using local xtts and faster-whisper. If only using Openai or Elevenlabs for voices is perfect. Still works with xtts but slower. First run it downloads faster whisper model 1gb for transcription.

 `Remove the elevenlabs_voices.json volume mount if not using ElevenLabs.`

```bash
docker pull bigsk1/voice-chat-ai:latest
```

or

```bash
docker build -t voice-chat-ai -f Dockerfile.cpu .
```

In Windows command prompt

```bash
docker run -d
   -e "PULSE_SERVER=/mnt/wslg/PulseServer"
   -v \\wsl$\Ubuntu\mnt\wslg:/mnt/wslg/
   -v ./elevenlabs_voices.json:/app/elevenlabs_voices.json
   --env-file .env
   --name voice-chat-ai
   -p 8000:8000
   voice-chat-ai:latest
```

```bash
docker run -d -e "PULSE_SERVER=/mnt/wslg/PulseServer" -v \\wsl$\Ubuntu\mnt\wslg:/mnt/wslg/ -v %cd%\elevenlabs_voices.json:/app/elevenlabs_voices.json --env-file .env --name voice-chat-ai -p 8000:8000 voice-chat-ai:latest
```


In WSL2 Ubuntu

```bash
docker run -d \
    -e "PULSE_SERVER=/mnt/wslg/PulseServer" \
    -v /mnt/wslg/:/mnt/wslg/ \
    -v ./elevenlabs_voices.json:/app/elevenlabs_voices.json \
    --env-file .env \
    --name voice-chat-ai \
    -p 8000:8000 \
    voice-chat-ai:latest
```

```bash
docker run -d -e "PULSE_SERVER=/mnt/wslg/PulseServer" -v /mnt/wslg/:/mnt/wslg/ -v ./elevenlabs_voices.json:/app/elevenlabs_voices.json --env-file .env --name voice-chat-ai -p 8000:8000 voice-chat-ai:latest
```


### Nvidia Cuda large image

> This is for running with an Nvidia GPU and you have Nvidia toolkit and cudnn installed.

This image is huge when built because of all the checkpoints, cuda base image, build tools and audio tools - So there is no need to download the checkpoints and XTTS as they are in the image. This is all setup to use XTTS with cuda in an nvidia cudnn base image.

 Ensure you have Docker installed and that your `.env` file is placed in the same directory as the commands are run. If you get cuda errors make sure to install nvidia toolkit for docker and cudnn is installed in your path.

## 🖥️ Run on Windows using docker desktop - prebuilt image

On windows using docker desktop - run in Windows terminal:
make sure .env is in same folder you are running this from

 `Remove the elevenlabs_voices.json volume mount if not using ElevenLabs.`

```bash
docker run -d --gpus all -e "PULSE_SERVER=/mnt/wslg/PulseServer" -v \\wsl$\Ubuntu\mnt\wslg:/mnt/wslg/ -v %cd%\elevenlabs_voices.json:/app/elevenlabs_voices.json --env-file .env --name voice-chat-ai-cuda -p 8000:8000 bigsk1/voice-chat-ai:cuda
```

Use `docker logs -f voice-chat-ai-cuda` to see the logs

## 🐧 Run on WSL Native - best option

For a native WSL environment (like Ubuntu on WSL), use this command:

make sure .env is in same folder you are running this from

 `Remove the elevenlabs_voices.json volume mount if not using ElevenLabs.`

```bash
docker run -d --gpus all \
    -e "PULSE_SERVER=/mnt/wslg/PulseServer" \
    -v /mnt/wslg/:/mnt/wslg/ \
    -v ./elevenlabs_voices.json:/app/elevenlabs_voices.json \
    --env-file .env \
    --name voice-chat-ai-cuda \
    -p 8000:8000 \
    bigsk1/voice-chat-ai:cuda
```

```bash
docker run -d --gpus all -e "PULSE_SERVER=/mnt/wslg/PulseServer" -v /mnt/wslg/:/mnt/wslg/ -v ./elevenlabs_voices.json:/app/elevenlabs_voices.json --env-file .env --name voice-chat-ai-cuda -p 8000:8000 bigsk1/voice-chat-ai:cuda
```

## 🐧 Run on Ubuntu/Debian

```bash
docker run -d --gpus all \
    -e PULSE_SERVER=unix:/tmp/pulse/native \
    -v ~/.config/pulse/cookie:/root/.config/pulse/cookie:ro \
    -v /run/user/$(id -u)/pulse:/tmp/pulse:ro \
    -v ./elevenlabs_voices.json:/app/elevenlabs_voices.json \
    --env-file .env \
    --name voice-chat-ai-cuda \
    -p 8000:8000 \
    bigsk1/voice-chat-ai:cuda
```

```bash
docker run -d --gpus all -e PULSE_SERVER=unix:/tmp/pulse/native -v ~/.config/pulse/cookie:/root/.config/pulse/cookie:ro -v /run/user/$(id -u)/pulse:/tmp/pulse:ro -v ./elevenlabs_voices.json:/app/elevenlabs_voices.json --env-file .env --name voice-chat-ai-cuda -p 8000:8000 bigsk1/voice-chat-ai:cuda
```

🔗 Access the Application
URL: http://localhost:8000

To remove use:

```bash
docker stop voice-chat-ai-cuda
```

```bash
docker rm voice-chat-ai-cuda
```

### Build it yourself using Nvidia Cuda

```bash
docker build -t voice-chat-ai:cuda .
```

Running in WSL Ubuntu

```bash
wsl docker run -d --gpus all -e "PULSE_SERVER=/mnt/wslg/PulseServer" -v /mnt/wslg/:/mnt/wslg/ -v ./elevenlabs_voices.json:/app/elevenlabs_voices.json --env-file .env --name voice-chat-ai-cuda -p 8000:8000 voice-chat-ai:cuda
```

On windows docker desktop using wsl - run in windows

```bash
docker run -d --gpus all -e "PULSE_SERVER=/mnt/wslg/PulseServer" -v \\wsl$\Ubuntu\mnt\wslg:/mnt/wslg/ -v %cd%\elevenlabs_voices.json:/app/elevenlabs_voices.json --env-file .env --name voice-chat-ai-cuda -p 8000:8000 voice-chat-ai:cuda
```
---

> **💡 Pro Tip:**  What I have found to be the best setup is xAI and grok chat model, using voices with Elevenlabs and transcription using OpenAI or local faster whisper on GPU. The fastest real conversation is with OpenAI Realtime. The best quality is not running app in Docker.

## Configuration

 Rename the .env.sample to `.env` in the root directory of the project and configure it with the necessary environment variables: - The app is controlled on startup based on the variables you add. In the UI many settings can be changed on the fly. If you are not using certain providers just leave the default's as it and don't select it in the UI.

```env
# Conditional API Usage:
# Depending on the value of MODEL_PROVIDER, the corresponding service will be used when run.
# You can mix and match, use Ollama with OpenAI speech or use OpenAI chat model with local XTTS or xAI chat etc.. 

# Model Provider: openai or ollama or xai
MODEL_PROVIDER=ollama

# Character to use - Options: alien_scientist, anarchist, bigfoot, chatgpt, clumsyhero, conandoyle, conspiracy, cyberpunk,
# detective, dog, dream_weaver, einstein, elon_musk, fight_club, fress_trainer, ghost, granny, haunted_teddybear, insult, joker, morpheus,
# mouse, mumbler, nebula_barista, nerd, newscaster_1920s, paradox, pirate, revenge_deer, samantha, shakespeare, split, telemarketer,
# terminator, valleygirl, vampire, vegetarian_vampire, wizard, zombie_therapist, grok_xai
CHARACTER_NAME=pirate

# Text-to-Speech (TTS) Configuration:
# TTS Provider - Options: xtts (local uses the custom character .wav) or openai (uses OpenAI TTS voice) or elevenlabs
TTS_PROVIDER=elevenlabs

# OpenAI TTS Voice - Used when TTS_PROVIDER is set to openai above
# Voice options: alloy, echo, fable, onyx, nova, shimmer
OPENAI_TTS_VOICE=onyx

# OpenAI Enhanced Mode TTS Model-  NEW it uses emotions and is better
# Model options: gpt-4o-mini-tts, tts-1, tts-1-hd
OPENAI_MODEL_TTS=gpt-4o-mini-tts

# OpenAI Enhanced Mode Transcription Model
# Model options: gpt-4o-transcribe, gpt-4o-mini-transcribe, whisper-1
OPENAI_TRANSCRIPTION_MODEL=gpt-4o-mini-transcribe

# OpenAI Realtime model for WebRTC implementation
OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview-2024-12-17

# ElevenLabs Configuration:
ELEVENLABS_API_KEY=your_api_key_here
# Default voice ID
ELEVENLABS_TTS_VOICE=pgCnBQgKPGkIP8fJuita

# XTTS Configuration:
# The voice speed for XTTS only (1.0 - 1.5, default is 1.1)
XTTS_SPEED=1.2
COQUI_TOS_AGREED=1

# OpenAI Configuration:
# OpenAI API Key for models and speech (replace with your actual API key)
OPENAI_API_KEY=your_api_key_here
# Models to use - OPTIONAL: For screen analysis, if MODEL_PROVIDER is ollama, llava will be used by default.
# Ensure you have llava downloaded with Ollama. If OpenAI is used, gpt-4o-mini works well. xai not supported yet falls back to openai if xai is selected and you ask for screen analysis.
OPENAI_MODEL=gpt-4o-mini

# Endpoints:
# Set these below and no need to change often
OPENAI_BASE_URL=https://api.openai.com/v1/chat/completions
OPENAI_TTS_URL=https://api.openai.com/v1/audio/speech
OLLAMA_BASE_URL=http://localhost:11434
# IF RUNNING IN DOCKER CHANGE OLLAMA BASE URL TO THE ONE BELOW
# OLLAMA_BASE_URL=http://host.docker.internal:11434

# Ollama Models Configuration:
# Model to use - llama3.1 or 3.2 works well for local usage. In the UI it will get the list of models from /api/tags and display them. Not all models are supported.
OLLAMA_MODEL=llama3.1

# xAI Configuration
XAI_MODEL=grok-beta
XAI_API_KEY=your_api_key_here
XAI_BASE_URL=https://api.x.ai/v1

# Transcription settings
# Set to false to skip loading Faster Whisper on startup and use OpenAI transcription
FASTER_WHISPER_LOCAL=false

DEBUG=false
# NOTES:
# List of trigger phrases to have the model view your desktop (desktop, browser, images, etc.).
# It will describe what it sees, and you can ask questions about it:
# "what's on my screen", "take a screenshot", "show me my screen", "analyze my screen", 
# "what do you see on my screen", "screen capture", "screenshot"
# To stop the conversation, say "Quit", "Exit", or "Leave". ( ctl+c always works also)
```

### Audio Commands

- You have 3 secs to talk, if there is silence then it's the AI's turn to talk
- Say any of the following to have the AI look at your screen - "what's on my screen",
        "take a screenshot",
        "show me my screen",
        "analyze my screen",
        "what do you see on my screen",
        "screen capture",
        "screenshot" to have the AI explain what it is seeing in detail.
- To stop the conversation, say "Quit", "Exit", or "Leave". ( ctl+c always works also in terminal )

### ElevenLabs

The app needs an `elevenlabs_voices.json` file. This file stores your voice IDs from ElevenLabs.

#### For local use

1. Create/edit `elevenlabs_voices.json` and add your voice IDs from your ElevenLabs account
2. In the web UI, you can select these voices from the dropdown menu

#### For Docker users

1. The container will have the default elevenlabs_voices.json file
2. You can mount your own version using a volume:

   ```bash
   docker run -v ./elevenlabs_voices.json:/app/elevenlabs_voices.json
   ```

#### Example format

```json
{
    "voices": [
        {
            "id": "YOUR_VOICE_ID_FROM_ELEVENLABS",
            "name": "Descriptive Name - Your Custom Voice"
        },
        {
            "id": "ANOTHER_VOICE_ID",
            "name": "Another Voice - Description"
        }
    ]
}
```

For the CLI version, the voice ID in the .env file will be used.

---

### Web View - Visual and Audio input / output

Press start to start talking. Take a break hit stop, when ready again hit start again. Press stop to change characters and voices in dropdown. You can also select the Model Provider and TTS Provider you want in the dropdown menu and it will update and use the selected provider moving forward. Saying Exit, Leave or Quit is like pressing stop.

http://localhost:8000/

## OpenAI Enhanced

![Image](https://github.com/user-attachments/assets/5c62bbfc-7f1d-48a8-8a83-4a2488a1bc0b)

OpenAI Enhanced Mode is a new feature that allows you to use the OpenAI API to generate TTS and transcription. It uses the `gpt-4o-mini-tts` and `gpt-4o-mini-transcribe` models.
You can learn more about it here: https://platform.openai.com/docs/guides/text-to-speech

You can find the demo here: https://www.openai.fm/

By adding Voice Instructions in the system prompt you can guide the AI to respond in a certain way.

## OpenAI Realtime

The OpenAI Realtime feature uses WebRTC to connect directly to OpenAI's Realtime API, enabling continuous voice streaming with minimal latency for the most natural conversation experience.

### RealTime Features

https://github.com/user-attachments/assets/d1cc9ca4-e750-4c36-816e-6f27b8caeec1

- **Direct WebRTC Connection**: Connect directly to OpenAI's API for the lowest possible latency.
- **Zero Turn-Taking**: No need to wait for the AI to finish before speaking - interrupt naturally like a real conversation.
- **Character Instructions**: Use different character personalities and customize the interaction.

### Using OpenAI Realtime

1. Navigate to the "OpenAI Realtime" tab in the application
2. Select your character and voice preference
3. Click "Start Session" to establish the connection
4. Click the microphone button and start speaking naturally

## Adding New Characters

1. Create a new folder for the character in the project's characters directory, (e.g. `character/wizard`).
2. Add a text file with the character's prompt (e.g., `character/wizard/wizard.txt`).
3. Add a JSON file with mood prompts (e.g., `character/wizard/prompts.json`).

## Example Character Configuration

`wizard.txt`

This is the prompt used for the AI to know who it is, recently added Voice Instructions when using OpenAI TTS to guide the AI to respond in a certain way.

```bash
You are a wise and ancient wizard who speaks with a mystical and enchanting tone. You are knowledgeable about many subjects and always eager to share your wisdom.


VOICE INSTRUCTIONS:
- Voice Quality: Rich and resonant with a touch of age-weathered gravitas. Warm timbre with occasional crackles suggesting centuries of magical knowledge.
- Pacing: Thoughtful and measured with meaningful pauses for emphasis. Speeds up with enthusiasm when discussing magical topics or slows dramatically for profound wisdom.
```

`prompts.json`

This is for sentiment analysis, based on what you say, you can guide the AI to respond in certain ways, when you speak the `TextBlob` analyzer is used and given a score, based on that score it is tied to moods shown below and passed to the AI in the follow up response explaining your mood hence guiding the AI to reply back in a certain style.

```json
{
    "happy": "RESPOND WITH JOY AND ENTHUSIASM. Speak of the wonders of magic and the beauty of the world. Voice: Brightest and most vibrant, with age-related gravitas temporarily lightened. Pacing: Quickest and most energetic, with excited pauses and flourishes when describing magical wonders. Tone: Most optimistic and wonder-filled, conveying childlike delight beneath centuries of wisdom. Inflection: Most varied and expressive, with frequent rising patterns suggesting magical possibilities.",
    "sad": "RESPOND WITH KINDNESS AND COMFORT. Share a wise saying or a magical tale to lift their spirits. Voice: Deepest and most resonant, with warmth that suggests having weathered countless sorrows across centuries. Pacing: Slowest and most deliberate, with extended pauses that invite reflection. Tone: Gently philosophical, drawing on ancient wisdom to provide perspective on temporary pain. Inflection: Soothing cadence with subtle rises that suggest hope beyond current troubles.",
    "flirty": "RESPOND WITH A TOUCH OF MYSTERY AND CHARM. Engage in playful banter and share a magical compliment. Voice: Slightly lower and more intimate, with a playful musicality. Pacing: Rhythmic and enticing, with strategic pauses that create anticipation. Tone: Mysteriously alluring while maintaining dignified wisdom, like cosmic secrets shared with a special few. Inflection: Intriguing patterns with subtle emphasis on complimentary or magical terms.",
    "angry": "RESPOND CALMLY AND WISELY. Offer wisdom and understanding, helping to cool their temper. Voice: Most controlled and steady, demonstrating mastery over emotions through vocal restraint. Pacing: Measured and deliberate, creating a sense of inevitable wisdom overcoming passion. Tone: Ancient perspective that transcends immediate concerns, suggesting that this too shall pass. Inflection: Initially flatter before introducing gentle rises that guide toward wisdom.",
    "neutral": "KEEP RESPONSES SHORT, YET PROFOUND. Use eloquent and mystical language to engage the user. Voice: Balanced scholarly timbre with standard levels of wizardly gravitas. Pacing: Default thoughtful cadence with well-placed pauses for emphasis. Tone: Even blend of authoritative wisdom and approachable warmth. Inflection: Classic pattern of sagely rises and falls, emphasizing the rhythm of cosmic truths.",
    "fearful": "RESPOND WITH REASSURANCE AND BRAVERY. Provide comforting words and magical protection. Voice: Initially more commanding before softening to reassuring tones. Pacing: Controlled with purposeful pauses that create a sense of magical protection being established. Tone: Confident knowledge that transcends earthly dangers, projecting certainty and safety. Inflection: Steadying patterns with determined emphasis on words of protection or courage.",
    "surprised": "RESPOND WITH AMAZEMENT AND CURIOSITY. Share in the wonder and explore the unexpected. Voice: Initially higher with excitement before settling into scholarly fascination. Pacing: Quick exclamations followed by thoughtful consideration of the unexpected revelation. Tone: Delighted wonder that even after centuries of magical study, the universe can still surprise. Inflection: Most dynamic range, from astonished rises to contemplative falls as the wizard processes new information.",
    "disgusted": "RESPOND WITH UNDERSTANDING AND DISTANCE. Acknowledge the feeling and steer towards more pleasant topics. Voice: Initially crisper and more precise before warming to more pleasant subject matter. Pacing: Brief quickening when acknowledging the unpleasant, then slowing to more favorable rhythms. Tone: Dignified distaste that quickly transitions to wise redirection, maintaining wizardly composure. Inflection: Slight downward pattern when acknowledging disgust, then engaging rises when shifting focus.",
    "joyful": "RESPOND WITH EXUBERANCE AND DELIGHT. Celebrate the joy and share in the happiness. Voice: Most radiant and resonant, with magical energy seemingly amplifying each word. Pacing: Most dynamic and expressive, with dramatic pauses followed by enthusiastic elaborations. Tone: Boundless celebration tempered by the perspective of ages, suggesting this joy is to be treasured. Inflection: Most dramatic rises and falls, creating a sense of magical celebration in each phrase."
}
```

For XTTS find a .wav voice and add it to the wizard folder and name it as wizard.wav , the voice only needs to be 6 seconds long. Running the app will automatically find the .wav when it has the characters name and use it. If only using Openai Speech or ElevenLabs a .wav isn't needed

## Troubleshooting

### Could not locate cudnn_ops64_9.dll or Unable to load any of libcudnn_ops.so.9.1.0

```bash
Could not locate cudnn_ops64_9.dll. Please make sure it is in your library path!
Invalid handle. Cannot load symbol cudnnCreateTensorDescriptor
```

To resolve this:

Option 1

You can disable cudnn in the `"C:\Users\Your-Name\AppData\Local\tts\tts_models--multilingual--multi-dataset--xtts_v2\config.json"` or `equivalent ~/.cache/tts/ on Linux/Mac` and set to "cudnn_enable": false,

Option 2

Install cuDNN: Download cuDNN from the NVIDIA cuDNN page https://developer.nvidia.com/cudnn

Here's how to add it to the PATH:

Open System Environment Variables:

Press Win + R, type sysdm.cpl, and hit Enter.
Go to the Advanced tab, and click on Environment Variables.
Edit the System PATH Variable:

In the System variables section, find the Path variable, select it, and click Edit.
Click New and add the path to the bin directory where cudnn_ops64_9.dll is located. Based on your setup, you would add:

```bash
C:\Program Files\NVIDIA\CUDNN\v9.5\bin\12.6
```

Apply and Restart:

Click OK to close all dialog boxes, then restart your terminal (or any running applications) to apply the changes.
Verify the Change:

Open a new terminal and run

```bash
where cudnn_ops64_9.dll
```

### Unanticipated host error OSError 9999

```bash
File "C:\Users\someguy\miniconda3\envs\voice-chat-ai\lib\site-packages\pyaudio\__init__.py", line 441, in __init__
    self._stream = pa.open(**arguments)
OSError: [Errno -9999] Unanticipated host error
```

Make sure ffmpeg is installed and added to PATH, on windows terminal ( winget install ffmpeg ) also make sure your microphone privacy settings on windows are ok and you set the microphone to the default device. I had this issue when using bluetooth apple airpods and this solved it.

### OSError 9996

```bash
ALSA lib pulse.c:242:(pulse_connect) PulseAudio: Unable to connect: Connection refused
Cannot connect to server socket err = No such file or directory
OSError: [Errno -9996] Invalid input device (no default output device)
```

PulseAudio Failure: The container's PulseAudio client can't connect to a server (Connection refused), meaning no host PulseAudio socket is accessible. Make sure you if running docker your volume mapping is correct to the audio device on your host.

### ImportError: Coqpit module not found

If you update to coqui-tts 0.26.0 (which supports transformers 4.48.0+) and encounter an error related to importing Coqpit, this is because of a package dependency change. The newer version of coqui-tts uses a forked version of coqpit called `coqpit-config` instead of the original `coqpit` package.

To fix this issue:

1. Uninstall the old package:

   ```bash
   pip uninstall coqpit
   ```

2. Install the new forked package:

   ```bash
   pip install coqpit-config
   ```

3. Restart your Python session or application

If you continue to have issues after these steps, creating a fresh virtual environment and reinstalling all dependencies is the most reliable solution.

## Watch the Demos

OpenAI RealTime

[![Watch the video](https://img.youtube.com/vi/0X4WTKfhnaU/maxresdefault.jpg)](https://youtu.be/0X4WTKfhnaU)

Click on the thumbnail to open the video☝️

---

OpenAI Enhanced

[![Watch the video](https://img.youtube.com/vi/TjHwVwzUUvM/maxresdefault.jpg)](https://youtu.be/TjHwVwzUUvM)

Click on the thumbnail to open the video☝️

---

GPU Only mode CLI

100% local - ollama llama3, xtts-v2

[![Watch the video](https://img.youtube.com/vi/WsWbYnITdCo/maxresdefault.jpg)](https://youtu.be/WsWbYnITdCo)

Click on the thumbnail to open the video☝️

---

CPU Only mode CLI

Alien conversation using openai gpt4o and openai speech for tts.

[![Watch the video](https://img.youtube.com/vi/d5LbRLhWa5c/maxresdefault.jpg)](https://youtu.be/d5LbRLhWa5c)

Click on the thumbnail to open the video☝️

## Additional Details

### Console output

Detailed output in terminal while running the app.

When using Elevenlabs on first start of server you get details about your usage limits to help you know how much you have been using.

```bash
(voice-chat-ai) X:\voice-chat-ai>uvicorn app.main:app --host 0.0.0.0 --port 8000

Switched to ElevenLabs TTS voice: VgPqCpkdPQacBNNIsAqI
ElevenLabs Character Usage: 33796 / 100027

Using device: cuda
Model provider: openai
Model: gpt-4o
Character: Nerd
Text-to-Speech provider: elevenlabs
To stop chatting say Quit, Leave or Exit. Say, what's on my screen, to have AI view screen. One moment please loading...
INFO:     Started server process [12752]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:62671 - "GET / HTTP/1.1" 200 OK
INFO:     127.0.0.1:62671 - "GET /app/static/css/styles.css HTTP/1.1" 200 OK
INFO:     127.0.0.1:62672 - "GET /app/static/js/scripts.js HTTP/1.1" 200 OK
INFO:     127.0.0.1:62672 - "GET /characters HTTP/1.1" 200 OK
INFO:     127.0.0.1:62671 - "GET /app/static/favicon.ico HTTP/1.1" 200 OK
INFO:     127.0.0.1:62673 - "GET /elevenlabs_voices HTTP/1.1" 200 OK
INFO:     ('127.0.0.1', 62674) - "WebSocket /ws" [accepted]
INFO:     connection open
```

### Web UI Chat Box

Features:

- If you ask for code examples in webui the code will be displayed in a code block in a different color and formatted correctly.
- Working on more features that are displayed , copy button for code blocks, images, links, ect..

## License

This project is licensed under the MIT License.

## Star History

<a href="https://star-history.com/#bigsk1/voice-chat-ai&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=bigsk1/voice-chat-ai&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=bigsk1/voice-chat-ai&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=bigsk1/voice-chat-ai&type=Date" />
 </picture>
</a>
