import pytest

from esopn.config import Settings


def test_capture_interval_is_validated() -> None:
    with pytest.raises(Exception):
        Settings(capture_interval=0.1)


def test_volume_range_is_validated() -> None:
    with pytest.raises(Exception):
        Settings(crowd_volume=1.5)


def test_mode_literal_validation() -> None:
    with pytest.raises(Exception):
        Settings(commentary_mode="invalid")
