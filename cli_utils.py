import os
import json
import base64
import asyncio
import aiohttp
from pathlib import Path
from typing import List
from PIL import ImageGrab

from core import synthesize_text

# Functions originally from app/app.py related to CLI and screenshot features

async def take_screenshot(temp_image_path: str) -> str:
    await asyncio.sleep(5)
    screenshot = ImageGrab.grab()
    screenshot = screenshot.resize((1024, 1024))
    screenshot.save(temp_image_path, 'JPEG')
    return temp_image_path

async def encode_image(image_path: str) -> str:
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

async def analyze_image(image_path: str, question_prompt: str) -> dict:
    encoded_image = await encode_image(image_path)
    api_url = 'https://api.openai.com/v1/chat/completions'
    headers = {'Authorization': f"Bearer {os.getenv('OPENAI_API_KEY')}", 'Content-Type': 'application/json'}
    payload = {
        'model': os.getenv('OPENAI_MODEL', 'gpt-4o'),
        'messages': [
            {'role': 'user', 'content': question_prompt},
            {'role': 'user', 'content': {'type': 'image_url', 'image_url': f'data:image/jpg;base64,{encoded_image}'}}
        ]
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()

async def execute_once(question_prompt: str) -> str:
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    temp_image_path = output_dir / 'temp_img.jpg'
    temp_audio_path = output_dir / 'temp_audio.wav'
    image_path = await take_screenshot(str(temp_image_path))
    response = await analyze_image(image_path, question_prompt)
    text_response = response.get('choices', [{}])[0].get('message', {}).get('content', 'No response received.')
    audio_bytes = await synthesize_text(text_response)
    with open(temp_audio_path, 'wb') as f:
        f.write(audio_bytes)
    return text_response
