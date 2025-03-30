"""
Shared resources used across the application
"""

import os
from dotenv import load_dotenv

load_dotenv()

# WebSocket clients
clients = set()
active_client_status = {}  # Track status of websocket clients

# Shared state variables
current_character = os.getenv("CHARACTER_NAME")  # Get from .env
conversation_active = False
conversation_history = []
continue_conversation = False  # Added missing variable

# Functions to get and set shared state
def get_current_character():
    """Get the current character."""
    global current_character # noqa: F824
    return current_character

def set_current_character(character):
    """Set the current character."""
    global current_character
    current_character = character

def is_conversation_active():
    """Check if a conversation is active."""
    global conversation_active # noqa: F824
    return conversation_active

def set_conversation_active(active):
    """Set the conversation active state."""
    global conversation_active
    conversation_active = active

def add_client(client):
    """Add a client to the set of connected clients."""
    clients.add(client)
    active_client_status[client] = True

def remove_client(client):
    """Remove a client from the set of connected clients."""
    clients.discard(client)
    if client in active_client_status:
        del active_client_status[client]

def is_client_active(client):
    """Check if a client is active."""
    return active_client_status.get(client, False)

def set_client_inactive(client):
    """Mark a client as inactive."""
    active_client_status[client] = False

def clear_conversation_history():
    """Clear the conversation history."""
    global conversation_history
    conversation_history = []
