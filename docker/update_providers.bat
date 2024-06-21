@echo off
if "%~2"=="" (
    echo Usage: %~0 ^<MODEL_PROVIDER^> ^<TTS_PROVIDER^>
    exit /b 1
)

set MODEL_PROVIDER=%1
set TTS_PROVIDER=%2

REM Update the .env file with the new model provider and TTS provider
powershell -Command "(gc .env) -replace '^MODEL_PROVIDER=.*', 'MODEL_PROVIDER=%MODEL_PROVIDER%' | Out-File -encoding ASCII .env"
powershell -Command "(gc .env) -replace '^TTS_PROVIDER=.*', 'TTS_PROVIDER=%TTS_PROVIDER%' | Out-File -encoding ASCII .env"

REM Restart the Docker container
docker restart voice-chat-ai


REM use   update_providers.bat ollama xtts    to update .env vaules in docker container after built
