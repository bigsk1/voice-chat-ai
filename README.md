
[![Python application](https://github.com/bigsk1/voice-chat-ai/actions/workflows/python-app.yml/badge.svg)](https://github.com/bigsk1/voice-chat-ai/actions/workflows/python-app.yml)

# Voice Chat AI

Voice Chat AI is a project that allows you to interact with different AI characters using speech. You can choose between various characters, each with unique personalities and voices. You can run all locally, you can use openai for chat and voice, you can mix between the two.

![Ai-Speech](https://imagedelivery.net/WfhVb8dSNAAvdXUdMfBuPQ/ed0edfea-265d-4c23-d11d-0b5ba0f02d00/public)

## Features

- **Supports both OpenAI and Ollama language models**: Choose the model that best fits your needs.
- **Provides text-to-speech synthesis using XTTS or OpenAI TTS**: Enjoy natural and expressive voices.
- **No typing needed, just speak!**: Hands-free interaction makes conversations smooth and effortless.
- **Analyzes user mood and adjusts AI responses accordingly**: Get personalized responses based on your mood.
- **You can, just by speaking, have the AI analyze your screen and chat about it**: Seamlessly integrate visual context into your conversations.
- **Easy configuration through environment variables**: Customize the application to suit your preferences with minimal effort.


## Installation

### Requirements

- Python 3.10
- CUDA-enabled GPU
- Microphone
- A sense of humor

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/bigsk1/voice-chat-ai.git
   cd voice-chat-ai
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\Activate`
   ```

   or use conda just make it python 3.10

   ```bash
   conda create --name voice-chat-ai python=3.10
   conda activate voice-chat-ai

   # Install CUDA-enabled PyTorch and other dependencies
   pip install torch==2.3.1+cu121 torchaudio==2.3.1+cu121 torchvision==0.18.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
   pip install -r requirements.txt

   # For CPU-only installations, use:
   pip install -r cpu_requirements.txt
   ```

3. Install dependencies:

   For GPU (CUDA) version:

   ```bash
   pip install -r requirements.txt
   ```

   For CPU-only version:

   ```bash
   pip install -r cpu_requirements.txt
   ```

### Download Checkpoints

You need to download the checkpoints for the models used in this project. You can download them from the GitHub releases page and extract the zip into the project folder.

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

#### Linux CLI Instructions

You can use the following commands to download and extract the files directly into the project directory:

```sh
# Navigate to the project directory
cd /path/to/your/voice-chat-ai

# Download and extract checkpoints.zip
wget https://github.com/bigsk1/voice-chat-ai/releases/download/models/checkpoints.zip
unzip checkpoints.zip -d .

# Download and extract XTTS-v2.zip
wget https://github.com/bigsk1/voice-chat-ai/releases/download/models/XTTS-v2.zip
unzip XTTS-v2.zip -d .
```

## Configuration

1. Rename the .env.sample to `.env` in the root directory of the project and configure it with the necessary environment variables: - The app is controlled based on the variables you add.

   ```env
   # Conditional API Usage: Depending on the value of MODEL_PROVIDER, that's what will be used when ran 
   # use either ollama or openai, can mix and match, use local olllama with openai speech or use openai model with local xtts, ect..

   # openai or ollama
   MODEL_PROVIDER=ollama

   # Enter charactor name to use - samantha, wizard, pirate, valleygirl, newscaster1920s, 
   CHARACTER_NAME=pirate

   # Text-to-Speech Provider - (xtts local uses the custom charactor .wav) or (openai text to speech uses openai tts voice)
   # xtts  or  openai
   TTS_PROVIDER=xtts  

   # The voice speed for xtts only ( 1.0 - 1.5 , default 1.1)
   XTTS_SPEED=1.1

   # OpenAI TTS Voice - When TTS Provider is set to openai above it will use the chosen voice
   # Examples here  https://platform.openai.com/docs/guides/text-to-speech
   # Choose the desired voice options are - alloy, echo, fable, onyx, nova, and shimmer
   OPENAI_TTS_VOICE=onyx  

   # SET THESE BELOW AND NO NEED TO CHANGE OFTEN #

   # Endpoints
   OPENAI_BASE_URL=https://api.openai.com/v1/chat/completions
   OPENAI_TTS_URL=https://api.openai.com/v1/audio/speech
   OLLAMA_BASE_URL=http://localhost:11434

   # OpenAI API Key for models and speech
   OPENAI_API_KEY=sk-11111111

   # Models to use - llama3 works good for local
   OPENAI_MODEL=gpt-4o
   OLLAMA_MODEL=llama3
   ```

## Usage

Run the application:

```bash
python app.py
```

### Commands

- To stop the conversation, say "Quit", "Exit", or "Leave".

## Adding New Characters

1. Create a new folder for the character in the project's characters directory.
2. Add a text file with the character's prompt (e.g., `wizard/wizard.txt`).
3. Add a JSON file with mood prompts (e.g., `wizard/prompts.json`).

## Example Character Configuration

`wizard/wizard.txt`

```
You are a wise and ancient wizard who speaks with a mystical and enchanting tone. You are knowledgeable about many subjects and always eager to share your wisdom.
```

`wizard/prompts.json`

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

For XTTS find a .wav voice and add it to the wizard folder and name it as wizard.wav , the voice only needs to be 6 seconds long. Running the app will automaticly find the .wav when it has the characters name and use it. 

## License

This project is licensed under the MIT License.
