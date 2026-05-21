# Contributing to Voice Chat AI

Thank you for your interest in this project. Voice Chat AI is maintained by a small team; clear expectations help everyone spend time on work that actually lands.

**Start here:** read the pinned guidance in [Discussion #55 — Adding Feature PR's](https://github.com/bigsk1/voice-chat-ai/discussions/55). This file expands on that post.

## Before you code

### New features or large changes

Please **do not** open large pull requests out of the blue.

1. Open a [GitHub Discussion](https://github.com/bigsk1/voice-chat-ai/discussions) or [issue](https://github.com/bigsk1/voice-chat-ai/issues) describing the idea.
2. Wait for maintainer feedback before investing in a full implementation.
3. If we agree on scope, mention the discussion/issue number in your PR.

We would rather say **no** (or **not now**) early than review half-integrated code that does not match how the app is run day to day.

### Bug fixes and small improvements

Pull requests are welcome without prior discussion when the change is narrow and obvious: a reproducible bug fix, typo/docs fix, or a small improvement that does not change architecture or dependencies.

Include steps to reproduce (for bugs) or how you tested the change.

## What fits this project (and what does not)

### LLM and TTS providers — curated, not exhaustive

The app already supports several **chat** providers (OpenAI, xAI, Ollama, Anthropic) and **TTS** options (OpenAI, xAI, ElevenLabs, Kokoro, Spark-TTS, Typecast). That set is intentional and sufficient for most users.

We are **not** looking to add every provider from the ecosystem (dozens of TTS backends, gateway wrappers, etc.). Each integration must be maintained, documented, and tested across Web UI, CLI, `.env` configuration, and install paths.

**Not soliciting right now:** broad gateway integrations (e.g. LiteLLM, OpenRouter) or similar “route to 100+ APIs” layers. Something like that *might* be considered in the future only if it is **complete** (see checklist below), does not burden all installs with heavy optional dependencies, and has been discussed and approved first. **Incomplete PRs that only touch `requirements.txt` or backend branches will be closed.**

### AI-assisted contributions

If you use AI tools to generate code, you are still responsible for the result. The PR must reflect **real** use of this application:

- Run and test **Web UI** and/or **CLI** as relevant — not only unit tests with mocked dependencies.
- Document new environment variables in `.env.sample`.
- Update user-facing docs (`README.md`, `INSTALL.md`) when behavior or setup changes.

## Pull request checklist

Use this before opening a PR (especially for features):

- [ ] Linked to an approved Discussion or issue (for new features).
- [ ] Tested locally: Web UI and/or CLI paths you changed.
- [ ] Updated `.env.sample` for any new config vars.
- [ ] Updated `README.md` / `INSTALL.md` if install or usage changed.
- [ ] **Dependencies updated in all required places** (see below).
- [ ] New `MODEL_PROVIDER` or `TTS_PROVIDER` values include UI parity where other providers have it (dropdowns, WebSocket handlers), not env-only stubs.

### Dependencies — all three, not just `requirements.txt`

This repo supports both **pip** (`requirements.txt`) and **uv** (`pyproject.toml` + `uv.lock`). Adding a Python package in only one file is incomplete.

If you add or change a dependency:

1. Add it to **`pyproject.toml`** (`[project].dependencies` or optional extras as appropriate).
2. Mirror it in **`requirements.txt`** (and any other requirements file you affect, e.g. `requirements_sparktts.txt` — see [README_REQUIREMENTS.md](README_REQUIREMENTS.md)).
3. Regenerate the lockfile: run **`uv lock`** and commit the updated **`uv.lock`**.

Call out in the PR description whether the new dependency is **required for all users** or only when a specific feature flag / env setting is enabled, and the install size impact if significant.

PyTorch remains a special case: install separately per [INSTALL.md](INSTALL.md); do not assume CUDA wheels belong in the core lockfile without discussion.

## How to run the project (for testing)

See [INSTALL.md](INSTALL.md) and [README.md](README.md). Typical paths:

- **Web UI:** run the FastAPI app and exercise voice chat, settings, and any provider you touched.
- **CLI:** `uv run python cli.py` (or your venv equivalent).

## Questions

- **Ideas / “would you want this?”** → [Discussions](https://github.com/bigsk1/voice-chat-ai/discussions)
- **Bugs** → [Issues](https://github.com/bigsk1/voice-chat-ai/issues) (use the bug template)

## License

By contributing, you agree that your contributions will be licensed under the same [MIT License](LICENSE) as the project.
