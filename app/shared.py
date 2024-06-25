import os
from dotenv import load_dotenv

load_dotenv()

clients = []
continue_conversation = False
conversation_history = []
_current_character = os.getenv("CHARACTER_NAME", "pirate")  # Default to "pirate" if not set in .env

def get_current_character():
    global _current_character
    return _current_character

def set_current_character(character):
    global _current_character
    _current_character = character
