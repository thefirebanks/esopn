from esopn.dialogue import ensure_speaker_tags, parse_dialogue_segments, split_speaker_lines


def test_ensure_speaker_tags_wraps_plain_text() -> None:
    assert ensure_speaker_tags("hello") == "[S1] hello"


def test_parse_dialogue_segments_parses_both_speakers() -> None:
    segments = parse_dialogue_segments("[S1] one [S2] two")
    assert segments == [("S1", "one"), ("S2", "two")]


def test_split_speaker_lines_returns_grouped_lines() -> None:
    alex, morgan = split_speaker_lines("[S1] first [S2] second [S1] third")
    assert alex == ["first", "third"]
    assert morgan == ["second"]
