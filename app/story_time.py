import json
import os
import re
from datetime import datetime, timedelta


DISPLAY_TIME_FORMAT = "%A, %B %d, %Y at %I:%M %p"
DISPLAY_TIME_PATTERN = r"[A-Za-z]+,\s+[A-Za-z]+\s+\d{2},\s+\d{4}\s+at\s+\d{2}:\d{2}\s+[AP]M"
HEADER_PATTERNS = (
    re.compile(rf"SCENE CLOCK:\s*({DISPLAY_TIME_PATTERN})"),
    re.compile(rf"LUCID INTERVAL OPENS:\s*({DISPLAY_TIME_PATTERN})"),
)
ARRIVAL_PATTERN = re.compile(rf"ARRIVAL:\s*({DISPLAY_TIME_PATTERN})")
HEADER_STRIP_PATTERNS = {
    "SCENE CLOCK:": re.compile(
        rf"^SCENE CLOCK:\s*{DISPLAY_TIME_PATTERN}(?P<trailing>[^\n]*)\n?",
        re.IGNORECASE,
    ),
    "ARRIVAL:": re.compile(
        rf"^ARRIVAL:\s*{DISPLAY_TIME_PATTERN}(?P<trailing>[^\n]*)\n?",
        re.IGNORECASE,
    ),
    "LUCID INTERVAL OPENS:": re.compile(
        rf"^LUCID INTERVAL OPENS:\s*{DISPLAY_TIME_PATTERN}(?P<trailing>[^\n]*)\n?",
        re.IGNORECASE,
    ),
}
OFFSET_PLACEHOLDER_PATTERN = re.compile(
    r"\{current_(?P<field>date_time|date|time|weekday|iso)_(?P<direction>minus|plus)_(?P<amount>\d+)_(?P<unit>minutes?|hours?|days?)\}"
)


def _join_preserved_header_tail(trailing, rest):
    trailing = trailing.strip()
    rest = rest.lstrip()
    if trailing and rest:
        return f"{trailing}\n{rest}"
    return trailing or rest


def format_display_time(dt):
    return dt.strftime(DISPLAY_TIME_FORMAT)


def _format_placeholder_value(field, dt):
    if field == "date_time":
        return dt.strftime(DISPLAY_TIME_FORMAT)
    if field == "date":
        return dt.strftime("%A, %B %d, %Y")
    if field == "time":
        return dt.strftime("%I:%M %p")
    if field == "weekday":
        return dt.strftime("%A")
    if field == "iso":
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    return None


def _placeholder_delta(amount, unit):
    amount = int(amount)
    if unit.startswith("minute"):
        return timedelta(minutes=amount)
    if unit.startswith("hour"):
        return timedelta(hours=amount)
    if unit.startswith("day"):
        return timedelta(days=amount)
    return timedelta()


def render_dynamic_prompt_template(template, now=None):
    if not template:
        return template

    now = now or datetime.now()

    def replace_offset(match):
        delta = _placeholder_delta(match.group("amount"), match.group("unit"))
        adjusted = now - delta if match.group("direction") == "minus" else now + delta
        return _format_placeholder_value(match.group("field"), adjusted) or match.group(0)

    rendered = OFFSET_PLACEHOLDER_PATTERN.sub(replace_offset, template)
    replacements = {
        "{current_date_time}": _format_placeholder_value("date_time", now),
        "{current_date}": _format_placeholder_value("date", now),
        "{current_time}": _format_placeholder_value("time", now),
        "{current_weekday}": _format_placeholder_value("weekday", now),
        "{current_iso}": _format_placeholder_value("iso", now),
    }

    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)

    return rendered


def parse_display_time(value):
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), DISPLAY_TIME_FORMAT)
    except ValueError:
        return None


def history_for_model(history):
    return [{"role": msg["role"], "content": msg["content"]} for msg in history]


def add_message_timestamp(message, when=None):
    stamped = dict(message)
    stamped.setdefault("timestamp_iso", (when or datetime.now()).isoformat(timespec="seconds"))
    return stamped


def _story_history_paths(characters_root, character_name):
    character_dir = os.path.join(characters_root, character_name)
    return (
        os.path.join(character_dir, "conversation_history.txt"),
        os.path.join(character_dir, "conversation_history.json"),
    )


def _parse_text_history(history_file):
    temp_history = []
    current_role = None
    current_content = ""

    with open(history_file, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("User:"):
                if current_role:
                    temp_history.append({"role": current_role, "content": current_content.strip()})
                current_role = "user"
                current_content = line[5:].strip()
            elif line.startswith("Assistant:"):
                if current_role:
                    temp_history.append({"role": current_role, "content": current_content.strip()})
                current_role = "assistant"
                current_content = line[10:].strip()
            else:
                current_content += "\n" + line

    if current_role:
        temp_history.append({"role": current_role, "content": current_content.strip()})

    return [_backfill_timestamp(msg) for msg in temp_history]


def _backfill_timestamp(message):
    stamped = dict(message)
    if stamped.get("timestamp_iso"):
        return stamped

    header_dt = extract_story_header_datetime(stamped.get("content", ""))
    if header_dt:
        stamped["timestamp_iso"] = header_dt.isoformat(timespec="seconds")
    return stamped


def load_story_history(characters_root, character_name):
    history_file, metadata_file = _story_history_paths(characters_root, character_name)

    if os.path.exists(metadata_file) and os.path.getsize(metadata_file) > 0:
        try:
            with open(metadata_file, "r", encoding="utf-8") as file:
                payload = json.load(file)
            if isinstance(payload, list):
                return [_backfill_timestamp(msg) for msg in payload if isinstance(msg, dict)]
        except (OSError, json.JSONDecodeError):
            pass

    if not os.path.exists(history_file) or os.path.getsize(history_file) == 0:
        return []

    return _parse_text_history(history_file)


def save_story_history(history, characters_root, character_name):
    history_file, metadata_file = _story_history_paths(characters_root, character_name)
    os.makedirs(os.path.dirname(history_file), exist_ok=True)

    normalized_history = [_backfill_timestamp(msg) for msg in history]

    with open(history_file, "w", encoding="utf-8") as file:
        for message in normalized_history:
            role = message["role"].capitalize()
            content = message["content"]
            file.write(f"{role}: {content}\n\n")

    with open(metadata_file, "w", encoding="utf-8") as file:
        json.dump(normalized_history, file, indent=2, ensure_ascii=True)

    return normalized_history


def extract_story_header_datetime(content):
    if not content:
        return None

    for pattern in HEADER_PATTERNS:
        match = pattern.search(content)
        if match:
            return parse_display_time(match.group(1))
    return None


def extract_arrival_datetime(content):
    if not content:
        return None
    match = ARRIVAL_PATTERN.search(content)
    if match:
        return parse_display_time(match.group(1))
    return None


def get_latest_assistant_timestamp(history):
    for message in reversed(history):
        if message.get("role") != "assistant":
            continue

        timestamp_iso = message.get("timestamp_iso")
        if timestamp_iso:
            try:
                return datetime.fromisoformat(timestamp_iso)
            except ValueError:
                pass

        parsed = extract_story_header_datetime(message.get("content", ""))
        if parsed:
            return parsed
    return None


def get_story_arrival_datetime(history):
    for message in history:
        if message.get("role") != "assistant":
            continue

        timestamp_iso = message.get("timestamp_iso")
        if timestamp_iso:
            try:
                return datetime.fromisoformat(timestamp_iso)
            except ValueError:
                pass

        parsed = extract_arrival_datetime(message.get("content", "")) or extract_story_header_datetime(
            message.get("content", "")
        )
        if parsed:
            return parsed
    return None


def humanize_timedelta(delta):
    total_seconds = max(0, int(delta.total_seconds()))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds and not parts:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

    return ", ".join(parts) if parts else "0 minutes"


def build_story_time_context(history, now):
    current_display = format_display_time(now)
    lines = [
        "SYSTEM TIME CONTEXT (APP-COMPUTED, AUTHORITATIVE):",
        f"- Real current wall-clock time for this response: {current_display}",
        f"- Real current ISO timestamp: {now.isoformat(timespec='seconds')}",
    ]

    previous_assistant = get_latest_assistant_timestamp(history)
    if previous_assistant:
        lines.append(
            f"- Previous assistant response wall-clock time: {format_display_time(previous_assistant)}"
        )
        lines.append(
            f"- Real elapsed time since the previous assistant response: {humanize_timedelta(now - previous_assistant)}"
        )
    else:
        lines.append("- This is the first assistant response in the story session.")

    arrival = get_story_arrival_datetime(history)
    if arrival:
        lines.append(f"- Frozen ARRIVAL/session-start time: {format_display_time(arrival)}")

    lines.extend(
        [
            "- Use these computed times as ground truth. Do not estimate elapsed time from the transcript alone.",
            "- If the story requires an exact timestamp header, print the real current wall-clock time above exactly.",
        ]
    )
    return "\n".join(lines)


def augment_story_system_message(system_message, history, now):
    return f"{system_message}\n\n{build_story_time_context(history, now)}"


def _strip_prefixed_story_headers(text, prefixes):
    remaining = text.lstrip()
    while True:
        stripped = False
        for prefix in prefixes:
            pattern = HEADER_STRIP_PATTERNS.get(prefix)
            match = pattern.match(remaining) if pattern else None
            if match:
                remaining = _join_preserved_header_tail(
                    match.group("trailing"),
                    remaining[match.end() :],
                )
                stripped = True
                break
            if remaining.startswith(prefix):
                newline = remaining.find("\n")
                remaining = "" if newline == -1 else remaining[newline + 1 :].lstrip()
                stripped = True
                break
        if not stripped:
            break
    return remaining


def enforce_story_response_headers(response_text, system_message, history, now):
    text = (response_text or "").strip()
    if not text:
        return text

    current_display = format_display_time(now)

    if "SCENE CLOCK:" in system_message:
        arrival = get_story_arrival_datetime(history) or now
        body = _strip_prefixed_story_headers(text, ("SCENE CLOCK:", "ARRIVAL:"))
        return (
            f"SCENE CLOCK: {current_display}\n"
            f"ARRIVAL: {format_display_time(arrival)}\n\n"
            f"{body.strip()}"
        ).strip()

    if "LUCID INTERVAL OPENS:" in system_message:
        body = _strip_prefixed_story_headers(text, ("LUCID INTERVAL OPENS:",))
        return f"LUCID INTERVAL OPENS: {current_display}\n\n{body.strip()}".strip()

    return response_text
