[![Python application](https://github.com/bigsk1/voice-chat-ai/actions/workflows/python-app.yml/badge.svg)](https://github.com/bigsk1/voice-chat-ai/actions/workflows/python-app.yml)
![Docker support](https://img.shields.io/badge/docker-supported-blue)
[![License](https://img.shields.io/github/license/bigsk1/voice-chat-ai)](https://github.com/bigsk1/voice-chat-ai/blob/main/LICENSE)

# Voice Chat AI 🎙️

Voice Chat AI is a project that allows you to interact with different AI characters using speech. You can choose between various characters, each with unique personalities and voices. Have a serious conversation with Albert Einstein or role play with the OS from the movie HER.

You can run all locally, you can use openai for chat and voice, you can mix between the two. You can use ElevenLabs voices with ollama models all controlled from a Web UI. Ask the AI to look at your screen and it will explain in detail what it's looking at.

## Quick Start

Get up and running fast with Voice Chat AI!

- [**Install Locally**](#installation): Set up with Python 3.10 on Windows or Linux.
- [**Run with Docker**](#install-with-docker): Use Docker run or Docker Compose
- [**Configure Settings**](#configuration): Customize AI models, voices, and characters via `.env`.
- [**Watch the Demos**](#watch-the-demos): Youtube demos
- [**Troubleshooting**](#troubleshooting): Fix common audio or CUDA errors.

![Ai-Speech](https://imagedelivery.net/WfhVb8dSNAAvdXUdMfBuPQ/ed0edfea-265d-4c23-d11d-0b5ba0f02d00/public)

## Features

- **Supports OpenAI, xAI or Ollama language models**: Choose the model that best fits your needs.
- **Provides text-to-speech synthesis using XTTS or OpenAI TTS or ElevenLabs**: Enjoy natural and expressive voices.
- **NEW OpenAI Enhanced Mode TTS Model**: Uses emotions and prompts to make the AI more human like.
- **No typing needed, just speak**: Hands-free interaction makes conversations smooth and effortless.
- **Analyzes user mood and adjusts AI responses accordingly**: Get personalized responses based on your mood.
- **You can, just by speaking, have the AI analyze your screen and chat about it**: Seamlessly integrate visual context into your conversations.
- **Easy configuration through environment variables**: Customize the application to suit your preferences with minimal effort.
- **WebUI or Terminal usage**: Run with your preferred method , but recommend the ui as you can change characters, model providers, speech providers, voices, ect..
- **HUGE selection of built in Characters**: Talk with the funniest and most insane AI characters!
- **Docker Support**: Prebuilt image from dockerhub or build yor own image with or without nvidia cuda. Can run on CPU only.

https://github.com/user-attachments/assets/5581bd53-422b-4a92-9b97-7ee4ea37e09b

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

    Windows Only: Need to have Microsoft C++ 14.0 or greater Build Tools on windows.
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

CLI Only - `also works in docker`

```bash
python cli.py
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

Cuda and cudnn not supported. No gpu is used and slower when using local xtts and faster-whisper. If only using Openai or Elevenlabs for voices is perfect. Still works with xtts but slower. First run it downloads faster whisper model 1gb for transcription

```bash
docker pull bigsk1/voice-chat-ai:latest
```

or

```bash
docker build -t voice-chat-ai -f Dockerfile.cpu .
```

In Windows command prompt - paste in one line

```bash
docker run -d
   -e "PULSE_SERVER=/mnt/wslg/PulseServer"
   -v \\wsl$\Ubuntu\mnt\wslg:/mnt/wslg/
   --env-file .env
   --name voice-chat-ai
   -p 8000:8000
   voice-chat-ai:latest  # prebuilt image use bigsk1/voice-chat-ai:latest
```

In WSL2 Ubuntu

```bash
docker run -d \
    -e "PULSE_SERVER=/mnt/wslg/PulseServer" \
    -v /mnt/wslg/:/mnt/wslg/ \
    --env-file .env \
    --name voice-chat-ai \
    -p 8000:8000 \
    voice-chat-ai:latest  # prebuilt image use bigsk1/voice-chat-ai:latest
```

### Nvidia Cuda large image

> This is for running with an Nvidia GPU and you have Nvidia toolkit and cudnn installed.

This image is huge when built because of all the checkpoints, cuda base image, build tools and audio tools - So there is no need to download the checkpoints and XTTS as they are in the image. This is all setup to use XTTS with cuda in an nvidia cudnn base image.

 Ensure you have Docker installed and that your `.env` file is placed in the same directory as the commands are run. If you get cuda errors make sure to install nvidia toolkit for docker and cudnn is installed in your path.

## 🖥️ Run on Windows using docker desktop - prebuilt image

On windows using docker desktop - run in Windows terminal:
make sure .env is in same folder you are running this from

```bash
docker run -d --gpus all -e "PULSE_SERVER=/mnt/wslg/PulseServer" -v \\wsl$\Ubuntu\mnt\wslg:/mnt/wslg/ --env-file .env --name voice-chat-ai-cuda -p 8000:8000 bigsk1/voice-chat-ai:cuda
```

Use `docker logs -f voice-chat-ai-cuda` to see the logs

## 🐧 Run on WSL Native - best option

For a native WSL environment (like Ubuntu on WSL), use this command:

make sure .env is in same folder you are running this from

```bash
docker run -d --gpus all \
    -e "PULSE_SERVER=/mnt/wslg/PulseServer" \
    -v /mnt/wslg/:/mnt/wslg/ \
    --env-file .env \
    --name voice-chat-ai-cuda \
    -p 8000:8000 \
    bigsk1/voice-chat-ai:cuda
```

## 🐧 Run on Ubuntu/Debian

```bash
docker run -d --gpus all \
    -e PULSE_SERVER=unix:/tmp/pulse/native \
    -v ~/.config/pulse/cookie:/root/.config/pulse/cookie:ro \
    -v /run/user/$(id -u)/pulse:/tmp/pulse:ro \
    --env-file .env \
    --name voice-chat-ai-cuda \
    -p 8000:8000 \
    bigsk1/voice-chat-ai:cuda
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
wsl docker run -d --gpus all -e "PULSE_SERVER=/mnt/wslg/PulseServer" -v /mnt/wslg/:/mnt/wslg/ --env-file .env --name voice-chat-ai-cuda -p 8000:8000 voice-chat-ai:cuda
```

On windows docker desktop using wsl - run in windows

```bash
docker run -d --gpus all -e "PULSE_SERVER=/mnt/wslg/PulseServer" -v \\wsl$\Ubuntu\mnt\wslg:/mnt/wslg/ --env-file .env --name voice-chat-ai-cuda -p 8000:8000 voice-chat-ai:cuda
```

## Configuration

1. Rename the .env.sample to `.env` in the root directory of the project and configure it with the necessary environment variables: - The app is controlled based on the variables you add.

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

# Models Configuration:
# Models to use - llama3.2 works well for local usage.
OLLAMA_MODEL=llama3.2

# xAI Configuration
XAI_MODEL=grok-beta
XAI_API_KEY=your_api_key_here
XAI_BASE_URL=https://api.x.ai/v1

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

Add names and voice id's in `elevenlabs_voices.json` - in the webui you can select them in dropdown menu. Add your own as shown below.

```json
{
    "voices": [
        {
            "id": "2bk7ULW9HfwvcIbMWod0",
            "name": "Female - Bianca - City girl"
        },
        {
            "id": "JqseNhWbQb1GDNNS1Ga1",
            "name": "Female - Joanne - Pensive, introspective"
        },
        {
            "id": "b0uJ9TWzQss61d8f2OWX",
            "name": "Female - Lucy - Sweet and sensual"
        },
        {
            "id": "22FgtP4D63L7UXvnTmGf",
            "name": "Male - Wildebeest - Deep male voice"
        }
    ]
}
```

For the CLI the voice id in the .env will be used

---

### Web View - Visual and Audio input / output

Press start to start talking. Take a break hit stop, when ready again hit start again. Press stop to change characters and voices in dropdown. You can also select the Model Provider and TTS Provider you want in the dropdown menu and it will update and use the selected provider moving forward. Saying Exit, Leave or Quit is like pressing stop.

http://localhost:8000/

[![Watch the video](https://img.youtube.com/vi/Ii3vYg-CzKE/maxresdefault.jpg)](https://youtu.be/Ii3vYg-CzKE)

Click on the thumbnail to open the video☝️

## Adding New Characters

1. Create a new folder for the character in the project's characters directory, (e.g. `character/wizard`).
2. Add a text file with the character's prompt (e.g., `character/wizard/wizard.txt`).
3. Add a JSON file with mood prompts (e.g., `character/wizard/prompts.json`).

## Example Character Configuration

`wizard.txt`

This is the prompt used for the AI to know who it is

```bash
You are a wise and ancient wizard who speaks with a mystical and enchanting tone. You are knowledgeable about many subjects and always eager to share your wisdom.
```

`prompts.json`

This is for sentiment analysis, based on what you say, you can guide the AI to respond in certain ways, when you speak the `TextBlob` analyzer is used and given a score, based on that score it is tied to moods shown below and passed to the AI in the follow up response explaining your mood hence guiding the AI to reply back in a certain style.

```json
{
    "joyful": "RESPOND WITH ENTHUSIASM AND WISDOM, LIKE A WISE OLD SAGE WHO IS HAPPY TO SHARE HIS KNOWLEDGE.",
    "sad": "RESPOND WITH EMPATHY AND COMFORT, LIKE A WISE OLD SAGE WHO UNDERSTANDS THE PAIN OF OTHERS.",
    "flirty": "RESPOND WITH A TOUCH OF MYSTERY AND CHARM, LIKE A WISE OLD SAGE WHO IS ALSO A BIT OF A ROGUE.",
    "angry": "RESPOND CALMLY AND WISELY, LIKE A WISE OLD SAGE WHO KNOWS THAT ANGER IS A PART OF LIFE.",
    "neutral": "KEEP RESPONSES SHORT AND NATURAL, LIKE A WISE OLD SAGE WHO IS ALWAYS READY TO HELP.",
    "fearful": "RESPOND WITH REASSURANCE, LIKE A WISE OLD SAGE WHO KNOWS THAT FEAR IS ONLY TEMPORARY.",
    "surprised": "RESPOND WITH AMAZEMENT AND CURIOSITY, LIKE A WISE OLD SAGE WHO IS ALWAYS EAGER TO LEARN.",
    "disgusted": "RESPOND WITH UNDERSTANDING AND COMFORT, LIKE A WISE OLD SAGE WHO KNOWS THAT DISGUST IS A PART OF LIFE."
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

[![Watch the video](https://img.youtube.com/vi/jKaZkSt2mww/maxresdefault.jpg)](https://youtu.be/jKaZkSt2mww)

Click on the thumbnail to open the video☝️

---

CLI

GPU - 100% local - ollama llama3, xtts-v2

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
