
# Voice Chat AI

Voice Chat AI is a project that allows you to interact with different AI characters using speech. You can choose between various characters, each with unique personalities and voices. You can run all locally, you can use openai for chat and voice, you can mix between the two.

![Ai-Speech](https://imagedelivery.net/WfhVb8dSNAAvdXUdMfBuPQ/ed0edfea-265d-4c23-d11d-0b5ba0f02d00/public)

## Features

- Supports both OpenAI and Ollama language models.
- Provides text-to-speech synthesis using XTTS or OpenAI TTS.
- Analyzes user mood and adjusts AI responses accordingly.
- Easy configuration through environment variables.

## Installation

### Requirements

- Python 3.10
- CUDA-enabled GPU
- Microphone
- A sence of humor! 

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

3. Install dependencies:


   For GPU (CUDA) version:

   ```bash
   pip install -r requirements.txt
   ```


   For CPU-only version:

   ```bash
   pip install -r cpu_requirements.txt
   ```

## Configuration

1. Create a `.env` file in the root directory of the project and configure it with the necessary environment variables:

   ```env
   MODEL_PROVIDER=ollama   # or openai
   TTS_PROVIDER=xtts       # or openai
   CHARACTER_NAME=samantha  # or any other character folder name

   OPENAI_API_KEY=your-openai-api-key
   OPENAI_MODEL=gpt-4o      # or your preferred OpenAI model
   OPENAI_BASE_URL=https://api.openai.com/v1/chat/completions
   OPENAI_TTS_URL=https://api.openai.com/v1/audio/speech
   OPENAI_TTS_VOICE=alloy  # or your preferred voice

   OLLAMA_MODEL=llama3     # or your preferred Ollama model
   OLLAMA_BASE_URL=http://localhost:11434
   ```


2. Add character-specific configuration files:
   - Create a folder named after your character (e.g., `samantha`).
   - Add a text file with the character's prompt (e.g., `samantha/samantha.txt`).
   - Add a JSON file with mood prompts (e.g., `samantha/prompts.json`).
   - Add the voice sample in the character folder (e.g., `samantha/samantha.wav`).


## Usage

Run the application:

```bash
python app.py
```

### Commands
- To stop the conversation, say "Quit", "Exit", or "Leave".

## Adding New Characters

1. Create a new folder for the character in the root directory.
2. Add a text file with the character's prompt (e.g., `wizard/wizard.txt`).
3. Add a JSON file with mood prompts (e.g., `wizard/prompts.json`).

## Example Character Configuration

### `wizard/wizard.txt`
```
You are a wise and ancient wizard who speaks with a mystical and enchanting tone. You are knowledgeable about many subjects and always eager to share your wisdom.
```

### `wizard/prompts.json`

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

## License

This project is licensed under the MIT License. 