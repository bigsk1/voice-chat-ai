"""
Stream helpers for handling responses from various LLM providers.
"""

import os
import json
import aiohttp
import logging

# Setup logger
logger = logging.getLogger(__name__)

# Global constants
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1/chat/completions')

async def stream_openai_response(character, prompt, model="gpt-4o-mini"):
    """
    Stream a response from OpenAI's API
    
    Args:
        character: The character responding
        prompt: The user's message
        model: The OpenAI model to use
    
    Yields:
        Chunks of text from the response as they arrive
    """
    if not OPENAI_API_KEY:
        logger.error("OpenAI API key not found")
        yield "I'm sorry, the OpenAI API key is missing. Please check your configuration."
        return

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Construct a simple system message
    system_message = f"You are {character}, a helpful AI assistant."
    
    # Build the messages payload
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": 1000
    }
    
    try:
        logger.info(f"Sending request to OpenAI with model {model}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                OPENAI_BASE_URL,
                headers=headers,
                json=payload,
                timeout=60
            ) as response:
                
                if response.status != 200:
                    error_message = await response.text()
                    logger.error(f"OpenAI API error: {response.status} - {error_message}")
                    yield f"Error from OpenAI API: {response.status}"
                    return
                
                # Process the streaming response
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    # Skip empty lines and "data: [DONE]"
                    if not line or line == "data: [DONE]":
                        continue
                    
                    # Remove "data: " prefix if present
                    if line.startswith("data: "):
                        line = line[6:]
                    
                    try:
                        chunk = json.loads(line)
                        delta_content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                        
                        if delta_content:
                            yield delta_content
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse chunk: {line}")
                        continue
                    
    except Exception as e:
        logger.error(f"Error streaming from OpenAI: {str(e)}")
        yield f"\nError streaming response: {str(e)}" 