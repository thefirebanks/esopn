"""Dialogue parsing helpers shared across commentary and TTS providers."""

import re

SPEAKER_TAG_PATTERN = re.compile(r"\[(S[12])\]\s*")


def ensure_speaker_tags(dialogue: str) -> str:
    """Ensure dialogue starts with an ESOPN speaker tag."""
    text = dialogue.strip()
    if text.startswith("[S1]") or text.startswith("[S2]"):
        return text

    match = re.search(r"\[S[12]\]", text)
    if match:
        return text[match.start() :]

    return f"[S1] {text}" if text else "[S1]"


def parse_dialogue_segments(dialogue: str) -> list[tuple[str, str]]:
    """Parse dialogue text into (speaker, text) segments."""
    parts = SPEAKER_TAG_PATTERN.split(dialogue)
    segments: list[tuple[str, str]] = []

    i = 1
    while i < len(parts) - 1:
        speaker = parts[i]
        text = parts[i + 1].strip()
        if text:
            segments.append((speaker, text))
        i += 2

    if not segments and dialogue.strip():
        segments.append(("S1", dialogue.strip()))

    return segments


def split_speaker_lines(dialogue: str) -> tuple[list[str], list[str]]:
    """Split dialogue into Alex and Morgan line lists."""
    alex_lines: list[str] = []
    morgan_lines: list[str] = []

    for speaker, text in parse_dialogue_segments(dialogue):
        if speaker == "S1":
            alex_lines.append(text)
        elif speaker == "S2":
            morgan_lines.append(text)

    return alex_lines, morgan_lines
