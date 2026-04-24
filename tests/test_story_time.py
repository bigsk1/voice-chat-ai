from datetime import datetime

from app.story_time import (
    add_message_timestamp,
    augment_story_system_message,
    enforce_story_response_headers,
    extract_arrival_datetime,
    extract_story_header_datetime,
    humanize_timedelta,
    render_dynamic_prompt_template,
)


def test_extract_story_header_datetime():
    content = "SCENE CLOCK: Thursday, April 23, 2026 at 04:48 AM\n\nTest"
    parsed = extract_story_header_datetime(content)
    assert parsed == datetime(2026, 4, 23, 4, 48)


def test_extract_arrival_datetime_from_inline_line():
    content = (
        "SCENE CLOCK: Thursday, April 23, 2026 at 04:49 AM\n"
        "ARRIVAL: Thursday, April 23, 2026 at 04:48 AM Too late? Not yet."
    )
    parsed = extract_arrival_datetime(content)
    assert parsed == datetime(2026, 4, 23, 4, 48)


def test_augment_story_system_message_includes_elapsed_gap():
    history = [
        add_message_timestamp(
            {
                "role": "assistant",
                "content": "SCENE CLOCK: Thursday, April 23, 2026 at 04:48 AM\nARRIVAL: Thursday, April 23, 2026 at 04:48 AM",
            },
            when=datetime(2026, 4, 23, 4, 48),
        )
    ]
    augmented = augment_story_system_message(
        "Base prompt",
        history,
        datetime(2026, 4, 23, 22, 48),
    )
    assert "18 hours" in augmented
    assert "Thursday, April 23, 2026 at 10:48 PM" in augmented


def test_enforce_story_response_headers_rewrites_scene_clock_and_arrival():
    history = [
        add_message_timestamp(
            {
                "role": "assistant",
                "content": "SCENE CLOCK: Thursday, April 23, 2026 at 04:48 AM\nARRIVAL: Thursday, April 23, 2026 at 04:48 AM",
            },
            when=datetime(2026, 4, 23, 4, 48),
        )
    ]
    fixed = enforce_story_response_headers(
        "SCENE CLOCK: Thursday, April 23, 2026 at 04:49 AM\nARRIVAL: Thursday, April 23, 2026 at 04:48 AM\n\nStill working.",
        "Prompt with SCENE CLOCK: and ARRIVAL:",
        history,
        datetime(2026, 4, 23, 22, 48),
    )
    assert fixed.startswith("SCENE CLOCK: Thursday, April 23, 2026 at 10:48 PM")
    assert "ARRIVAL: Thursday, April 23, 2026 at 04:48 AM" in fixed
    assert fixed.endswith("Still working.")


def test_enforce_story_response_headers_preserves_inline_arrival_narrative():
    history = []
    fixed = enforce_story_response_headers(
        "SCENE CLOCK: Friday, April 24, 2026 at 12:11 AM\n"
        "ARRIVAL: Friday, April 24, 2026 at 12:11 AM The coffee is cold. Daisy hops down.\n\n"
        "QUICK STATUS:\n- MINUTES ON SCENE: 0",
        "Prompt with SCENE CLOCK: and ARRIVAL:",
        history,
        datetime(2026, 4, 24, 0, 11),
    )
    assert "The coffee is cold. Daisy hops down." in fixed
    assert "QUICK STATUS:" in fixed


def test_enforce_story_response_headers_preserves_single_line_double_header_body():
    history = []
    fixed = enforce_story_response_headers(
        "SCENE CLOCK: Friday, April 24, 2026 at 12:11 AM ARRIVAL: Friday, April 24, 2026 at 12:11 AM The coffee is cold.",
        "Prompt with SCENE CLOCK: and ARRIVAL:",
        history,
        datetime(2026, 4, 24, 0, 11),
    )
    assert fixed.endswith("The coffee is cold.")


def test_humanize_timedelta_handles_hours_and_minutes():
    assert humanize_timedelta(datetime(2026, 4, 23, 22, 48) - datetime(2026, 4, 23, 4, 49)) == "17 hours, 59 minutes"


def test_render_dynamic_prompt_template_supports_relative_minutes():
    rendered = render_dynamic_prompt_template(
        "Call came in at {current_time_minus_22_minutes}; arrival is {current_time}.",
        now=datetime(2026, 4, 24, 1, 7),
    )
    assert rendered == "Call came in at 12:45 AM; arrival is 01:07 AM."


def test_render_dynamic_prompt_template_supports_relative_iso_days():
    rendered = render_dynamic_prompt_template(
        "Tomorrow marker: {current_iso_plus_1_day}.",
        now=datetime(2026, 4, 24, 1, 7),
    )
    assert rendered == "Tomorrow marker: 2026-04-25T01:07:00."
