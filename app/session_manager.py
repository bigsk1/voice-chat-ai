"""
Session manager for the voice chat application.
Handles tracking of user sessions, character assignments, and model overrides.
"""

import os
import logging
from .shared import get_current_character

# Setup logger
logger = logging.getLogger(__name__)

# Global variables
MODEL_PROVIDER = os.getenv('MODEL_PROVIDER', 'openai')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

# Session storage
sessions = {}
session_history = {}
session_providers = {}
session_models = {}

class SessionManager:
    """Manages user sessions and conversation history."""
    
    def __init__(self):
        """Initialize the session manager."""
        self.sessions = {}
        self.session_history = {}
    
    def get_character_for_session(self, client_id):
        """Get the character assigned to a session."""
        if not client_id:
            return get_current_character()
            
        if client_id in self.sessions:
            return self.sessions[client_id]
        
        # Default to the current character
        return get_current_character()
    
    def set_character_for_session(self, client_id, character):
        """Assign a character to a session."""
        if not client_id:
            return
            
        self.sessions[client_id] = character
        logger.info(f"Set character {character} for session {client_id}")
    
    def add_to_history(self, character, role, message):
        """Add a message to the conversation history for a character."""
        if not character in self.session_history:
            self.session_history[character] = []
            
        self.session_history[character].append({
            "role": "user" if role.lower() == "you" else "assistant",
            "content": message
        })
        
        # Limit history size
        if len(self.session_history[character]) > 30:
            self.session_history[character] = self.session_history[character][-30:]

def get_provider_from_session(client_id):
    """Get the model provider for a session."""
    if client_id and client_id in session_providers:
        return session_providers[client_id]
    return MODEL_PROVIDER

def get_openai_model_from_session(client_id):
    """Get the OpenAI model for a session."""
    if client_id and client_id in session_models:
        return session_models[client_id]
    return OPENAI_MODEL

def set_provider_for_session(client_id, provider):
    """Set the model provider for a session."""
    if not client_id:
        return
    session_providers[client_id] = provider

def set_model_for_session(client_id, model):
    """Set the model for a session."""
    if not client_id:
        return
    session_models[client_id] = model

# Create a singleton instance
session_manager = SessionManager() 