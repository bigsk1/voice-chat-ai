# Dynamic Prompt Templating and Per-Character TTS Filter

Two small systems that together let a character prompt be **time-aware** and **spoken selectively** — the LLM can see structured data every turn while the TTS only reads the immersive narrative aloud.

---

## 1. Dynamic Prompt Templating

### What it does

Before every chat call, the character's `story_<name>.txt` (or `<name>.txt`) prompt file is run through a small template renderer that substitutes **real-world date/time placeholders** into the system prompt.

This means the LLM sees the *actual current moment* at the start of every single turn — not just on first load.

### Why it exists

A character prompt that is read from disk once at startup quickly becomes stale. The LLM has no built-in clock. Without the renderer, there is no honest way for a story to ask:

- "How much real-world time has passed since the last turn?"
- "It's 11 PM on a Tuesday — change the mood."
- "You've been in a coma for six weeks since the attack on `{current_date}`."
- "The bomb was called in 22 minutes before you arrived at `{current_date_time}`."

With the renderer, every assistant turn starts with fresh wall-clock values, and stories can compare the **current** timestamp to a **previous** timestamp they themselves wrote into the conversation history on the prior turn. That diff is real elapsed time — the core mechanic behind `story_lucid_intervals` (coma time-gaps in days/weeks/months) and `story_bomb_threat` (bomb countdown in seconds/minutes).

### Supported placeholders

Use any of these directly in a character `.txt` prompt file. They get substituted every turn before the prompt reaches the model.

| Placeholder            | Example output                            | Notes                               |
| ---------------------- | ----------------------------------------- | ----------------------------------- |
| `{current_date_time}`  | `Monday, April 22, 2026 at 03:14 PM`      | Full human-readable timestamp.      |
| `{current_date}`       | `Monday, April 22, 2026`                  | Weekday + date, no time.            |
| `{current_time}`       | `03:14 PM`                                | 12-hour clock.                      |
| `{current_weekday}`    | `Monday`                                  | Weekday name only.                  |
| `{current_iso}`        | `2026-04-22T15:14:00`                     | Machine-friendly ISO 8601 (local).  |

Placeholders that are not present in the file are left alone. Placeholders you do not recognize (typos) are passed through as-is.

### Where it lives

`render_prompt_template(template)` is defined in both `app/app.py` and `cli.py`. It is called on the raw prompt text **every turn**, inside:

- `app/app.py` — the CLI-attached main loop (`user_chatbot_conversation`).
- `app/app_logic.py` — `process_text()` (the FastAPI handler powering the default WebUI).
- `app/enhanced_logic.py` — the "enhanced" OpenAI-realtime-style loop.
- `cli.py` — standalone CLI entry point.

All four paths re-read and re-render the template on every turn, so the time stays accurate even in very long sessions.

### How to use it in your own character

Just drop a placeholder anywhere in your `story_<name>.txt`:

```
It is currently {current_date_time}. This is the real-world time of RIGHT NOW.
Every time the player speaks to you, compare this timestamp to the previous
assistant message's timestamp header and narrate the elapsed time honestly.
```

Then have the assistant **echo the timestamp back** in each of its replies, so the next turn has something to compare against. The existing pattern in `story_lucid_intervals` and `story_bomb_threat` is:

```
LUCID INTERVAL OPENS: {current_date_time}
...
```

or

```
SCENE CLOCK: {current_date_time}
ARRIVAL: {current_date_time (frozen from turn one)}
...
```

The assistant is instructed to:

1. Print the current timestamp header as the first line of every reply.
2. Read the **previous** assistant message from history for its timestamp header.
3. Compute the delta.
4. Drive in-story consequences off the delta (coma passage of weeks, bomb ticking down, seasons changing, etc.).

### Gotchas

- The renderer uses local system time (`datetime.now()`), not UTC. Fine for single-user local setups; if you ever run this server-side for multiple users in different timezones, swap it for a tz-aware variant.
- The LLM can see the current timestamp, but only retains awareness of past timestamps if the **assistant** wrote them into its own replies. User messages are not time-stamped automatically.
- If you use tight time math (like the bomb clock), remember that **TTS generation + playback + user thinking** is real elapsed time between turns. On slow TTS providers this can be 1–3 minutes per turn. Your story prompt should acknowledge that reality rather than fight it.

---

## 2. Per-Character TTS Filter (`tts_filter.json`)

### What it does

Each character folder (`characters/<name>/`) may optionally contain a `tts_filter.json` file that controls what gets **spoken aloud** versus what is only **kept in conversation history and shown in the UI**.

It is a thin regex-based gate applied to each assistant response right before the text is sent to TTS.

### Why it exists

Time-aware and structured stories often need the LLM to output machine-readable headers, status blocks, and meta lines — for example:

```
SCENE CLOCK: Thursday, April 23, 2026 at 04:12 AM
ARRIVAL: Thursday, April 23, 2026 at 04:12 AM

I'm stepping out of Truck 3. Daisy knocks her head against my good knee...

QUICK STATUS:
- MINUTES ON SCENE: 3
- Pulse: 72 -> 84

1. Approach locker 47.
2. Call for the disruptor.
3. Do it your own way.
```

All of that is valuable:

- `SCENE CLOCK` is how the **next** turn computes the real elapsed time.
- The delta status block is how the LLM tracks vitals across turns.
- The UI displays the whole thing so you can *see* the game state.

But having TTS read "SCENE CLOCK Thursday April twenty-third twenty twenty-six at four twelve AY EM" before every narrative beat is immersion-breaking. Same for listing the entire status block out loud each turn.

The TTS filter lets you hide those lines from speech **without** hiding them from the model or the UI.

### File location

```
characters/<character_name>/tts_filter.json
```

Missing file = no filter is applied. Full back-compat for every existing character. A malformed file logs a warning and is treated as if it were missing.

### Schema

```json
{
    "_comment": "Anything with an underscore prefix is ignored by the loader.",

    "strip_line_patterns": [
        "^\\s*SCENE CLOCK\\b",
        "^\\s*ARRIVAL\\b"
    ],

    "strip_patterns": [
        "\\[META:.*?\\]"
    ],

    "replace": [
        { "pattern": "\\bEOD\\b", "with": "E O D" },
        { "pattern": "\\bK9\\b",  "with": "K-9" },
        { "pattern": "\\bANFO\\b", "with": "an-foe" }
    ]
}
```

All three keys are optional. Any you omit is treated as empty. All regexes are compiled **case-insensitive**.

| Key                    | What it does                                                                 | Typical use                                                                 |
| ---------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `strip_line_patterns`  | Regex matched against each **line**. Any matching line is dropped entirely.  | Timestamp headers, status block headers, bullet-list meta rows.             |
| `strip_patterns`       | Regex removed as a substring (can span lines; `DOTALL` is on).               | Inline meta tags, bracketed director's notes, internal tool-call artifacts. |
| `replace`              | List of `{pattern, with}` pairs applied as regex substitutions.              | Pronunciation normalization — spelling out acronyms, fixing weird words.    |

Invalid regex patterns are logged as warnings and ignored; the rest of the file still works.

### Pipeline

```
                 chatbot_response  (raw from LLM)
                         │
          ┌──────────────┴──────────────┐
          │                             │
display_response                apply_tts_filter(raw, character)
(unfiltered,                             │
 format_story_response_text)             │
          │                              ▼
          │                      sanitize_response
          │                              │
          ▼                              ▼
conversation_history          process_and_play (TTS)
      + WebUI
      + conversation_history.txt
```

Key property: **the filter runs BEFORE `sanitize_response`.** This matters because `sanitize_response` strips colons, brackets, and hyphens (its regex is `[^\w\s,.\'!?]` for non-xAI providers). If we filtered after sanitize, a pattern like `^SCENE CLOCK:` would never match because the `:` would already be gone. By filtering first we match against the LLM's actual structured output.

The opposite-direction guarantee is equally important: **`display_response` is never filtered.** It stores the full, structured, timestamped text in `conversation_history.txt` so the LLM has everything it needs on the next turn, and the UI displays it verbatim.

### Example — `story_bomb_threat`

`characters/story_bomb_threat/tts_filter.json`:

```json
{
    "strip_line_patterns": [
        "^\\s*SCENE CLOCK\\b",
        "^\\s*ARRIVAL\\b"
    ],
    "strip_patterns": [],
    "replace": [
        { "pattern": "\\bEOD\\b",  "with": "E O D" },
        { "pattern": "\\bK9\\b",   "with": "K-9" },
        { "pattern": "\\bIED\\b",  "with": "I E D" },
        { "pattern": "\\bUSMC\\b", "with": "U S M C" },
        { "pattern": "\\bANFO\\b", "with": "an-foe" },
        { "pattern": "\\bFD\\b",   "with": "Fire Department" },
        { "pattern": "\\bPD\\b",   "with": "Police Department" }
    ]
}
```

**Before** filter (what the LLM produces and what history stores):

```
SCENE CLOCK: Thursday, April 23, 2026 at 04:12 AM
ARRIVAL: Thursday, April 23, 2026 at 04:12 AM

I'm stepping out of Truck 3. My EOD instincts haven't left, and this K9
is better than any IED-sniffer the USMC gave me.
```

**After** filter (what goes to TTS):

```
I'm stepping out of Truck 3. My E O D instincts haven't left, and this K-9
is better than any I E D-sniffer the U S M C gave me.
```

### Tuning tips

- Start minimal. Strip just the headers you find intrusive, see how it sounds, iterate.
- If you want the `QUICK STATUS` block silent too, add `"^\\s*QUICK STATUS\\b"` plus a pattern for its bullet rows, e.g. `"^\\s*-\\s+(MINUTES ON SCENE|BOMB CLOCK|Pulse|Hands|Bomb suit)\\b"`.
- Use `replace` aggressively for acronyms. TTS models vary wildly in how they pronounce `EOD`, `K9`, `HVAC`, etc. — normalize them here instead of fighting the model.
- Avoid stripping the **numbered options** block (`^\d+\.\s`). The player usually needs to hear them to make a choice.

### Caching and hot-reload

The filter file is cached in-memory per process, keyed by its modification time. That means you can:

1. Start a session.
2. Edit `tts_filter.json` in your editor.
3. The **next** turn picks up the new rules with no restart.

Great for iterating on immersion tuning without losing conversation state.

### Where it's wired

The helper pair `load_tts_filter()` / `apply_tts_filter()` is defined in `app/app.py` and mirrored in `cli.py`. It is wired into:

- `app/app.py` — main `user_chatbot_conversation` loop.
- `app/app_logic.py` — `process_text()` (default WebUI).
- `app/enhanced_logic.py` — enhanced realtime loop (here it also fixed a pre-existing bug where sanitized text was being stored in history; now the raw/display text goes to history and UI, filtered+sanitized text goes to TTS).
- `cli.py` — CLI main loop.

---

## 3. How the two features work together

The dynamic prompt renderer makes the LLM **time-aware**. The TTS filter makes the resulting structured output **immersive to listen to**. Neither depends on the other, but they are most useful as a pair:

- The renderer lets the LLM write `SCENE CLOCK: <now>` as the first line of every turn.
- The filter silences that line from TTS while keeping it in history for next-turn time-delta math.
- Players hear pure narrative.
- The LLM still has every structured anchor it needs to keep the world's clock ticking honestly.

This is the backbone of the real-time-clock mechanic in `story_bomb_threat` and the long-form time-gap mechanic in `story_lucid_intervals`, and the same two primitives let you build any time-aware or structured-state story on top of the standard character/prompts.json system without touching the core Python.
