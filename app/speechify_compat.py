"""Small helpers for talking to the Speechify API."""

SPEECHIFY_CALLER = "voice-chat-ai"
# Keep in step with the version in pyproject.toml.
SPEECHIFY_CALLER_VERSION = "1.4.1"


def speechify_caller_headers():
    """Attribution headers that tell Speechify which integration sent a request."""
    return {
        "Speechify-Caller": SPEECHIFY_CALLER,
        "Speechify-Caller-Version": SPEECHIFY_CALLER_VERSION,
    }
