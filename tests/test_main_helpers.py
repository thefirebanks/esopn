from pathlib import Path
from types import SimpleNamespace

import pytest

from esopn.main import _resolve_gemini_api_key, _validate_output_path


def test_resolve_gemini_api_key_prefers_settings_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "legacy-value")
    settings = SimpleNamespace(gemini_api_key="settings-value")
    assert _resolve_gemini_api_key(settings) == "settings-value"


def test_resolve_gemini_api_key_uses_legacy_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "legacy-value")
    settings = SimpleNamespace(gemini_api_key="")
    assert _resolve_gemini_api_key(settings) == "legacy-value"


def test_validate_output_path_rejects_paths_outside_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "outside.wav"
    with pytest.raises(ValueError):
        _validate_output_path(outside)


def test_validate_output_path_allows_paths_inside_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    inside = Path("artifacts") / "test.wav"
    resolved = _validate_output_path(inside)
    assert resolved == (tmp_path / "artifacts" / "test.wav").resolve()
