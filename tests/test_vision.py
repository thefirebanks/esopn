from esopn.vision import VisionAnalyzer


def test_parse_response_extracts_first_valid_json_object() -> None:
    analyzer = VisionAnalyzer.__new__(VisionAnalyzer)
    text = (
        "noise before {not valid} and then "
        '{"action":"A","details":["d"],"mood":"progress","intensity":6}'
        " trailing"
    )

    result = analyzer._parse_response(text)
    assert result.action == "A"
    assert result.mood == "progress"
    assert result.intensity == 6


def test_parse_response_handles_empty_text() -> None:
    analyzer = VisionAnalyzer.__new__(VisionAnalyzer)
    result = analyzer._parse_response("")
    assert result.action == "Analysis in progress"
