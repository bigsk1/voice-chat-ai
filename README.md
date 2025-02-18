[![Python application](https://github.com/bigsk1/voice-chat-ai/actions/workflows/python-app.yml/badge.svg)](https://github.com/bigsk1/voice-chat-ai/actions/workflows/python-app.yml)
![Docker support](https://img.shields.io/badge/docker-supported-blue)
[![License](https://img.shields.io/github/license/bigsk1/voice-chat-ai)](https://github.com/bigsk1/voice-chat-ai/blob/main/LICENSE)

# Voice Chat AI 🎙️

Voice Chat AI is a project that allows you to interact with different AI characters using speech. You can choose between various characters, each with unique personalities and voices. Have a serious conversation with Albert Einstein or role play with the OS from the movie HER. 

You can run all locally, you can use openai for chat and voice, you can mix between the two. You can use ElevenLabs voices with ollama models all controlled from a Web UI. Ask the AI to look at your screen and it will explain in detail what it's looking at. 

![Ai-Speech](https://imagedelivery.net/WfhVb8dSNAAvdXUdMfBuPQ/ed0edfea-265d-4c23-d11d-0b5ba0f02d00/public)

## Features

- **Supports OpenAI, xAI or Ollama language models**: Choose the model that best fits your needs.
- **Provides text-to-speech synthesis using XTTS or OpenAI TTS or ElevenLabs**: Enjoy natural and expressive voices.
- **No typing needed, just speak**: Hands-free interaction makes conversations smooth and effortless.
- **Analyzes user mood and adjusts AI responses accordingly**: Get personalized responses based on your mood.
- **You can, just by speaking, have the AI analyze your screen and chat about it**: Seamlessly integrate visual context into your conversations.
- **Easy configuration through environment variables**: Customize the application to suit your preferences with minimal effort.
- **WebUI or Terminal usage**: Run with your preferred method , but recommend the ui as you can change characters, model providers, speech providers, voices, ect..
- **HUGE selection of built in Characters**: Talk with the funniest and most insane AI characters!


https://github.com/user-attachments/assets/5581bd53-422b-4a92-9b97-7ee4ea37e09b


## Installation

### Requirements

- Python 3.10
- CUDA-enabled GPU
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
   
    For CPU-only version: clone the cpu-only branch
   https://github.com/bigsk1/voice-chat-ai/tree/cpu-only


2. Create a virtual environment: 🐍

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\Activate`
   ```

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
   
   pip install -r requirements.txt
   ```

   For CPU-only version (No UI) : clone the cpu-only branch
   https://github.com/bigsk1/voice-chat-ai/tree/cpu-only


    Make sure you have ffmpeg downloaded, on windows terminal ( winget install ffmpeg ) or checkout https://ffmpeg.org/download.html then restart shell or vscode, type ffmpeg -version to see if installed correctly

    Local XTTS you also might need cuDNN for using nvidia GPU https://developer.nvidia.com/cudnn  and make sure C:\Program Files\NVIDIA\CUDNN\v9.5\bin\12.6
is in system PATH or whatever version you downloaded

### Optional - Download Checkpoints - ONLY IF YOU ARE USING THE LOCAL TTS

If you are only using speech with Openai or Elevenlabs then you don't need this. To use the local TTS download the checkpoints for the models used in this project ( the docker image has the local xtts and checkpoints in it already ). You can download them from the GitHub releases page and extract the zip and put into the project folder.

- [Download Checkpoint](https://github.com/bigsk1/voice-chat-ai/releases/download/models/checkpoints.zip)
- [Download XTTS-v2](https://github.com/bigsk1/voice-chat-ai/releases/download/models/XTTS-v2.zip)

After downloading, place the folders as follows:

```bash
voice-chat-ai/
├── checkpoints/
│   ├── base_speakers/
│   │   ├── EN/
│   │   │   └── checkpoint.pth
│   │   ├── ZH/
│   │   │   └── checkpoint.pth
│   ├── converter/
│   │   └── checkpoint.pth
├── XTTS-v2/
│   ├── config.json
│   ├── other_xtts_files...
```

## Usage

Run the application: 🏃

Web UI
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Find on http://localhost:8000/


CLI Only

```bash
python cli.py
```

## Docker - large image - Experimental!

This is for running with an Nvidia GPU and you have Nvidia toolkit and cudnn installed. 

This image is huge when built because of all the checkpoints, cuda base image, build tools and audio tools - So there is no need to download the checkpoints and XTTS as they are in the image. This is all setup to use XTTS, if your not using XTTS for speech it should still work but it is just a large docker image and will take awhile, if you don't want to deal with that then run the app natively or build your own image without the xtts and checkpoints folders, if you are not using the local TTS.

This guide will help you quickly set up and run the **Voice Chat AI** Docker container. Ensure you have Docker installed and that your `.env` file is placed in the same directory as the commands are run. If you get cuda errors make sure to install nvidia toolkit for docker and cudnn is installed in your path.

---

## 📄 Prerequisites
1. Docker installed on your system.
2. A `.env` file in the same folder as the `docker run` command. This file should contain all necessary environment variables for the application.

---

## 🖥️ Run on Windows using docker desktop - prebuilt image
On windows using docker desktop - run in Windows terminal:
make sure .env is in same folder you are running this from
```bash
docker run -d --gpus all
   -e "PULSE_SERVER=/mnt/wslg/PulseServer"
   -v \\wsl$\Ubuntu\mnt\wslg:/mnt/wslg/
   --env-file .env
   --name voice-chat-ai
   -p 8000:8000
   bigsk1/voice-chat-ai:latest
```

Use `docker logs -f voice-chat-ai` to see the logs

## 🐧 Run on WSL Native - best option
For a native WSL environment (like Ubuntu on WSL), use this command:

make sure .env is in same folder you are running this from

```bash
docker run -d --gpus all \
    -e "PULSE_SERVER=/mnt/wslg/PulseServer" \
    -v /mnt/wslg/:/mnt/wslg/ \
    --env-file .env \
    --name voice-chat-ai \
    -p 8000:8000 \
    bigsk1/voice-chat-ai:latest
```

## 🐧 Run on Ubuntu/Debian

```bash
docker run -d --gpus all \
    -e PULSE_SERVER=unix:/tmp/pulse/native \
    -v ~/.config/pulse/cookie:/root/.config/pulse/cookie:ro \
    -v /run/user/$(id -u)/pulse:/tmp/pulse:ro \
    --env-file .env \
    --name voice-chat-ai \
    -p 8000:8000 \
    bigsk1/voice-chat-ai:latest
```
🔗 Access the Application
URL: http://localhost:8000

To remove use: 

```bash
docker stop voice-chat-ai
```

```bash
docker rm voice-chat-ai
```

## Build it yourself with cuda: 

```bash
docker build -t voice-chat-ai .
```
On windows docker desktop using wsl - run in windows

```bash
wsl docker run -d --gpus all -e "PULSE_SERVER=/mnt/wslg/PulseServer" -v /mnt/wslg/:/mnt/wslg/ --env-file .env --name voice-chat-ai -p 8000:8000 voice-chat-ai:latest
```

Running from wsl

```bash
docker run -d --gpus all -e "PULSE_SERVER=/mnt/wslg/PulseServer" -v \\wsl$\Ubuntu\mnt\wslg:/mnt/wslg/ --env-file .env --name voice-chat-ai -p 8000:8000 voice-chat-ai:latest
```

## Docker build without local xtts and no cuda

```bash
docker build -t voice-chat-ai-no-xtts -f no-xtts-Dockerfile .
```

In Windows command prompt

```bash
docker run -d
   -e "PULSE_SERVER=/mnt/wslg/PulseServer"
   -v \\wsl$\Ubuntu\mnt\wslg:/mnt/wslg/
   --env-file .env
   --name voice-chat-ai-no-xtts
   -p 8000:8000
   voice-chat-ai-no-xtts:latest
```

In WSL2 Ubuntu 

```bash
docker run -d \
    -e "PULSE_SERVER=/mnt/wslg/PulseServer" \
    -v /mnt/wslg/:/mnt/wslg/ \
    --env-file .env \
    --name voice-chat-ai-no-xtts \
    -p 8000:8000 \
    voice-chat-ai-no-xtts:latest
```

## Configuration ⚙️

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

Add names and voice id's in `elevenlabs_voices.json` - in the webui you can select them in dropdown menu.

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
            "id": "2pF3fJJNnWg1nDwUW5CW",
            "name": "Male - Eustis - Fast speaking"
        },
        {
            "id": "pgCnBQgKPGkIP8fJuita",
            "name": "Male - Jarvis - Tony Stark AI"
        },
        {
            "id": "kz8mB8WAwV9lZ0fuDqel",
            "name": "Male - Nigel - Mysterious intriguing"
        },
        {
            "id": "MMHtVLagjZxJ53v4Wj8o",
            "name": "Male - Paddington - British narrator"
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

```
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

Install cuDNN: Download cuDNN from the NVIDIA cuDNN page https://developer.nvidia.com/cudnn

Here’s how to add it to the PATH:

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

### Unanticipated host error

```bash
File "C:\Users\someguy\miniconda3\envs\voice-chat-ai\lib\site-packages\pyaudio\__init__.py", line 441, in __init__
    self._stream = pa.open(**arguments)
OSError: [Errno -9999] Unanticipated host error
```

Make sure ffmpeg is installed and added to PATH, on windows terminal ( winget install ffmpeg ) also make sure your microphone privacy settings on windows are ok and you set the microphone to the default device. I had this issue when using bluetooth apple airpods and this solved it.

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
