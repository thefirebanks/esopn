import numpy as np

from esopn.crowd import CrowdAudio, CrowdManager, CrowdReaction
from esopn.sfx import SFXManager


def test_sfx_then_commentary_resamples_to_commentary_rate() -> None:
    manager = SFXManager(sample_rate=44100, enabled=True)
    sfx = manager.generator.generate_ding(duration=0.1)
    commentary = np.zeros(24000, dtype=np.float32)

    mixed = manager.create_sfx_then_commentary(
        sfx,
        commentary,
        commentary_sample_rate=24000,
        gap_duration=0.1,
    )
    assert len(mixed) > len(commentary)


def test_crowd_mix_handles_sample_rate_mismatch() -> None:
    commentary = np.ones(24000, dtype=np.float32) * 0.2
    crowd_wave = np.ones(44100, dtype=np.float32) * 0.3
    crowd_audio = CrowdAudio(
        audio=crowd_wave,
        sample_rate=44100,
        duration=1.0,
        reaction=CrowdReaction.MURMUR,
    )
    manager = CrowdManager(sample_rate=44100)

    mixed = manager.mix_with_commentary(commentary, 24000, crowd_audio, crowd_position="under")
    assert len(mixed) == len(commentary)
    assert mixed.dtype == np.float32
