"""Sound effects module - generates synthetic SFX for coding events.

Sound effects are triggered by detected events and play BEFORE commentary
for dramatic impact (like sports broadcasts: horn → "TOUCHDOWN!").
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .audio_utils import resample_audio

logger = logging.getLogger(__name__)


class SFXType(Enum):
    """Types of sound effects."""

    # Success sounds
    AIR_HORN = "air_horn"  # Tests pass, build success
    DING = "ding"  # Commit success, small wins
    VICTORY_FANFARE = "victory"  # Major milestone

    # Failure sounds
    SAD_TROMBONE = "sad_trombone"  # Tests fail
    GLASS_BREAK = "glass_break"  # Error/Exception
    BUZZER = "buzzer"  # Build failure

    # WWE sounds
    BELL_RING = "bell_ring"  # Match start/end moment
    DRAMATIC_HIT = "dramatic_hit"  # Big moment impact

    # Neutral/transition
    WHOOSH = "whoosh"  # Fast transition
    CODE_DIFF = "code_diff"  # Code changes/diffs visible


@dataclass
class SoundEffect:
    """Container for a sound effect."""

    audio: np.ndarray
    sample_rate: int
    duration: float
    sfx_type: SFXType


class SFXGenerator:
    """Generates synthetic sound effects using signal processing."""

    def __init__(self, sample_rate: int = 44100, volume: float = 0.8):
        """
        Initialize SFX generator.

        Args:
            sample_rate: Audio sample rate
            volume: Base volume level (0.0 to 1.0)
        """
        self.sample_rate = sample_rate
        self.volume = volume
        self._rng = np.random.default_rng()

    def _apply_envelope(
        self,
        audio: np.ndarray,
        attack: float = 0.01,
        decay: float = 0.1,
        sustain: float = 0.7,
        release: float = 0.2,
    ) -> np.ndarray:
        """Apply ADSR envelope to audio."""
        samples = len(audio)
        envelope = np.ones(samples, dtype=np.float32)

        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate)
        release_samples = int(release * self.sample_rate)

        # Attack
        if attack_samples > 0 and attack_samples < samples:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

        # Decay
        decay_end = attack_samples + decay_samples
        if decay_samples > 0 and decay_end < samples:
            envelope[attack_samples:decay_end] = np.linspace(1, sustain, decay_samples)

        # Release
        if release_samples > 0 and release_samples < samples:
            envelope[-release_samples:] = np.linspace(sustain, 0, release_samples)

        return audio * envelope

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent clipping."""
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio

    # =========================================================================
    # SUCCESS SOUNDS
    # =========================================================================

    def generate_air_horn(self, duration: float = 0.8) -> SoundEffect:
        """Generate air horn sound - tests pass, build success!"""
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)

        # Air horn is multiple harmonics with slight detuning
        base_freq = 350  # Base frequency

        # Main tone with harmonics
        horn = np.zeros(samples, dtype=np.float32)
        for i, harmonic in enumerate([1, 2, 3, 4, 5]):
            # Slight random detune for each harmonic
            detune = self._rng.uniform(-5, 5)
            freq = base_freq * harmonic + detune
            amplitude = 1.0 / (i + 1)  # Higher harmonics quieter
            horn += np.sin(2 * np.pi * freq * t) * amplitude

        # Add some "brassy" noise
        noise = self._rng.standard_normal(samples) * 0.05
        horn = horn + noise.astype(np.float32)

        # Envelope - quick attack, sustained, quick release
        horn = self._apply_envelope(horn, attack=0.02, decay=0.05, sustain=0.9, release=0.1)

        # Normalize and apply volume
        horn = self._normalize(horn) * self.volume

        return SoundEffect(
            audio=horn.astype(np.float32),
            sample_rate=self.sample_rate,
            duration=duration,
            sfx_type=SFXType.AIR_HORN,
        )

    def generate_ding(self, duration: float = 0.5) -> SoundEffect:
        """Generate pleasant ding sound - commit success, small wins."""
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)

        # Bell-like sound with harmonics
        base_freq = 880  # A5

        ding = np.zeros(samples, dtype=np.float32)
        # Fundamental and harmonics (bell-like spectrum)
        for harmonic, amp in [(1, 1.0), (2, 0.5), (3, 0.3), (4.2, 0.2), (5.4, 0.1)]:
            freq = base_freq * harmonic
            ding += np.sin(2 * np.pi * freq * t) * amp

        # Exponential decay for bell-like sound
        decay = np.exp(-4 * t).astype(np.float32)
        ding = ding * decay

        # Normalize and apply volume
        ding = self._normalize(ding) * self.volume * 0.7  # Slightly quieter

        return SoundEffect(
            audio=ding.astype(np.float32),
            sample_rate=self.sample_rate,
            duration=duration,
            sfx_type=SFXType.DING,
        )

    def generate_victory_fanfare(self, duration: float = 1.2) -> SoundEffect:
        """Generate short victory fanfare - major milestones!"""
        samples = int(duration * self.sample_rate)
        np.linspace(0, duration, samples)

        fanfare = np.zeros(samples, dtype=np.float32)

        # Three ascending notes (like a trumpet fanfare)
        notes = [
            (0.0, 0.3, 440),  # A4
            (0.25, 0.3, 554),  # C#5
            (0.5, 0.6, 659),  # E5 (longer final note)
        ]

        for start, note_dur, freq in notes:
            start_sample = int(start * self.sample_rate)
            note_samples = int(note_dur * self.sample_rate)
            end_sample = min(start_sample + note_samples, samples)

            note_t = np.linspace(0, note_dur, end_sample - start_sample)

            # Brass-like tone
            note = np.sin(2 * np.pi * freq * note_t)
            note += np.sin(2 * np.pi * freq * 2 * note_t) * 0.5
            note += np.sin(2 * np.pi * freq * 3 * note_t) * 0.25

            # Note envelope
            note_env = np.ones(len(note))
            attack = int(0.02 * self.sample_rate)
            release = int(0.1 * self.sample_rate)
            if attack < len(note):
                note_env[:attack] = np.linspace(0, 1, attack)
            if release < len(note):
                note_env[-release:] = np.linspace(1, 0, release)

            note = note * note_env
            fanfare[start_sample:end_sample] += note.astype(np.float32)

        # Normalize and apply volume
        fanfare = self._normalize(fanfare) * self.volume

        return SoundEffect(
            audio=fanfare.astype(np.float32),
            sample_rate=self.sample_rate,
            duration=duration,
            sfx_type=SFXType.VICTORY_FANFARE,
        )

    # =========================================================================
    # FAILURE SOUNDS
    # =========================================================================

    def generate_sad_trombone(self, duration: float = 1.5) -> SoundEffect:
        """Generate sad trombone (wah wah waaah) - tests fail."""
        samples = int(duration * self.sample_rate)
        np.linspace(0, duration, samples)

        trombone = np.zeros(samples, dtype=np.float32)

        # Three descending notes with vibrato
        notes = [
            (0.0, 0.35, 293),  # D4
            (0.35, 0.35, 261),  # C4
            (0.7, 0.75, 220),  # A3 (longer, lower)
        ]

        for start, note_dur, base_freq in notes:
            start_sample = int(start * self.sample_rate)
            note_samples = int(note_dur * self.sample_rate)
            end_sample = min(start_sample + note_samples, samples)

            note_t = np.linspace(0, note_dur, end_sample - start_sample)

            # Add pitch bend down for "wah" effect
            pitch_bend = np.linspace(1.0, 0.95, len(note_t))
            freq = base_freq * pitch_bend

            # Cumulative phase for smooth frequency changes
            phase = np.cumsum(2 * np.pi * freq / self.sample_rate)

            # Trombone-like tone
            note = np.sin(phase)
            note += np.sin(2 * phase) * 0.4  # 2nd harmonic
            note += np.sin(3 * phase) * 0.2  # 3rd harmonic

            # Add vibrato
            vibrato = np.sin(2 * np.pi * 5 * note_t) * 0.02
            note = note * (1 + vibrato)

            # Note envelope with slow attack (muted trombone feel)
            note_env = np.ones(len(note))
            attack = int(0.05 * self.sample_rate)
            release = int(0.15 * self.sample_rate)
            if attack < len(note):
                note_env[:attack] = np.linspace(0, 1, attack)
            if release < len(note):
                note_env[-release:] = np.linspace(1, 0.3, release)

            note = note * note_env
            trombone[start_sample:end_sample] += note.astype(np.float32)

        # Normalize and apply volume
        trombone = self._normalize(trombone) * self.volume

        return SoundEffect(
            audio=trombone.astype(np.float32),
            sample_rate=self.sample_rate,
            duration=duration,
            sfx_type=SFXType.SAD_TROMBONE,
        )

    def generate_glass_break(self, duration: float = 0.6) -> SoundEffect:
        """Generate glass breaking sound - errors/exceptions."""
        samples = int(duration * self.sample_rate)

        # Glass break is mostly high-frequency noise with sharp attack
        noise = self._rng.standard_normal(samples).astype(np.float32)

        # High-pass filter effect (emphasize high frequencies)
        # Simple difference filter
        filtered = np.zeros(samples, dtype=np.float32)
        filtered[1:] = noise[1:] - 0.9 * noise[:-1]

        # Add some "tinkle" tones
        t = np.linspace(0, duration, samples)
        tinkle_freqs = [2000, 3000, 4000, 5000, 6000]
        tinkle = np.zeros(samples, dtype=np.float32)
        for freq in tinkle_freqs:
            # Random decay rate for each frequency
            decay_rate = self._rng.uniform(8, 15)
            amp = self._rng.uniform(0.1, 0.3)
            tinkle += np.sin(2 * np.pi * freq * t) * np.exp(-decay_rate * t) * amp

        # Combine noise and tinkle
        glass = filtered * 0.6 + tinkle * 0.4

        # Sharp attack envelope
        envelope = np.exp(-5 * t).astype(np.float32)
        glass = glass * envelope

        # Normalize and apply volume
        glass = self._normalize(glass) * self.volume

        return SoundEffect(
            audio=glass.astype(np.float32),
            sample_rate=self.sample_rate,
            duration=duration,
            sfx_type=SFXType.GLASS_BREAK,
        )

    def generate_buzzer(self, duration: float = 0.5) -> SoundEffect:
        """Generate buzzer sound - build failure, wrong answer."""
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)

        # Harsh buzzer sound - low frequency with distortion
        base_freq = 100

        # Square wave approximation (harsh sound)
        buzzer = np.sign(np.sin(2 * np.pi * base_freq * t))
        # Add some harmonics
        buzzer += np.sign(np.sin(2 * np.pi * base_freq * 2 * t)) * 0.5

        # Add noise for extra harshness
        noise = self._rng.standard_normal(samples) * 0.2
        buzzer = buzzer + noise.astype(np.float32)

        # Envelope
        buzzer = self._apply_envelope(buzzer, attack=0.01, decay=0.05, sustain=0.8, release=0.1)

        # Normalize and apply volume (slightly quieter - it's harsh)
        buzzer = self._normalize(buzzer) * self.volume * 0.6

        return SoundEffect(
            audio=buzzer.astype(np.float32),
            sample_rate=self.sample_rate,
            duration=duration,
            sfx_type=SFXType.BUZZER,
        )

    # =========================================================================
    # WWE SOUNDS
    # =========================================================================

    def generate_bell_ring(self, duration: float = 1.0) -> SoundEffect:
        """Generate wrestling bell ring - match moments!"""
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)

        # Bell sound - multiple inharmonic partials
        base_freq = 600

        bell = np.zeros(samples, dtype=np.float32)
        # Bell partials (slightly inharmonic for metallic sound)
        partials = [(1.0, 1.0), (2.0, 0.6), (3.0, 0.4), (4.2, 0.25), (5.4, 0.15)]

        for partial, amp in partials:
            freq = base_freq * partial
            # Each partial has different decay
            decay = np.exp(-(3 + partial) * t)
            bell += np.sin(2 * np.pi * freq * t) * amp * decay

        # Add initial "strike" transient
        strike_samples = int(0.01 * self.sample_rate)
        if strike_samples < samples:
            strike = self._rng.standard_normal(strike_samples) * 0.3
            strike_env = np.exp(-np.linspace(0, 10, strike_samples))
            bell[:strike_samples] += (strike * strike_env).astype(np.float32)

        # Normalize and apply volume
        bell = self._normalize(bell) * self.volume

        return SoundEffect(
            audio=bell.astype(np.float32),
            sample_rate=self.sample_rate,
            duration=duration,
            sfx_type=SFXType.BELL_RING,
        )

    def generate_dramatic_hit(self, duration: float = 0.4) -> SoundEffect:
        """Generate dramatic impact sound - big WWE moments!"""
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)

        # Low frequency thump
        thump_freq = 60
        thump = np.sin(2 * np.pi * thump_freq * t)
        thump_decay = np.exp(-8 * t)
        thump = thump * thump_decay

        # Mid-range crack
        crack_freq = 200
        crack = np.sin(2 * np.pi * crack_freq * t)
        crack += np.sin(2 * np.pi * crack_freq * 2 * t) * 0.5
        crack_decay = np.exp(-12 * t)
        crack = crack * crack_decay

        # High-frequency noise burst
        noise = self._rng.standard_normal(samples) * 0.3
        noise_decay = np.exp(-20 * t)
        noise = noise * noise_decay

        # Combine
        hit = thump * 0.5 + crack * 0.3 + noise.astype(np.float32) * 0.2

        # Normalize and apply volume
        hit = self._normalize(hit) * self.volume

        return SoundEffect(
            audio=hit.astype(np.float32),
            sample_rate=self.sample_rate,
            duration=duration,
            sfx_type=SFXType.DRAMATIC_HIT,
        )

    # =========================================================================
    # TRANSITION SOUNDS
    # =========================================================================

    def generate_whoosh(self, duration: float = 0.3) -> SoundEffect:
        """Generate whoosh sound - fast transitions."""
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)

        # Filtered noise with pitch sweep
        noise = self._rng.standard_normal(samples)

        # Create pitch sweep using amplitude modulation at varying frequency
        sweep_freq = np.linspace(500, 2000, samples)  # Rising pitch
        sweep = np.sin(2 * np.pi * sweep_freq * t / self.sample_rate * np.arange(samples))

        # Modulate noise with sweep
        whoosh = noise * np.abs(sweep) * 0.5

        # Envelope - builds up then fades
        envelope = np.sin(np.pi * t / duration)  # Sine curve envelope
        whoosh = whoosh * envelope

        # Normalize and apply volume
        whoosh = self._normalize(whoosh.astype(np.float32)) * self.volume * 0.5

        return SoundEffect(
            audio=whoosh.astype(np.float32),
            sample_rate=self.sample_rate,
            duration=duration,
            sfx_type=SFXType.WHOOSH,
        )

    def generate_code_diff(self, duration: float = 0.4) -> SoundEffect:
        """Generate code diff sound - like shuffling/swapping code blocks.

        A satisfying "chunk-whoosh" sound that evokes code being swapped out,
        like shuffling cards or a mechanical switch.
        """
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)

        # Part 1: Initial "chunk" - mechanical click/switch sound (first 30%)
        chunk_samples = int(samples * 0.3)
        chunk_t = t[:chunk_samples]

        # Low frequency thump for the "chunk"
        chunk_freq = 150
        chunk = np.sin(2 * np.pi * chunk_freq * chunk_t)
        chunk += np.sin(2 * np.pi * chunk_freq * 2 * chunk_t) * 0.5
        # Sharp attack, quick decay
        chunk_env = np.exp(-15 * chunk_t)
        chunk = chunk * chunk_env

        # Add some click noise
        click_noise = self._rng.standard_normal(chunk_samples) * 0.3
        click_env = np.exp(-30 * chunk_t)
        chunk = chunk + (click_noise * click_env).astype(np.float32)

        # Part 2: Swoosh/shuffle sound (remaining 70%)
        swoosh_samples = samples - chunk_samples
        swoosh_t = np.linspace(0, duration * 0.7, swoosh_samples)

        # Frequency sweep down (like something sliding into place)
        sweep_freq = np.linspace(800, 200, swoosh_samples)
        phase = np.cumsum(2 * np.pi * sweep_freq / self.sample_rate)
        swoosh = np.sin(phase) * 0.4

        # Add filtered noise for "paper shuffle" texture
        noise = self._rng.standard_normal(swoosh_samples)
        # Simple high-pass effect
        filtered = np.zeros(swoosh_samples, dtype=np.float32)
        filtered[1:] = (noise[1:] - 0.7 * noise[:-1]).astype(np.float32)
        swoosh = swoosh + filtered * 0.3

        # Envelope for swoosh - quick attack from chunk, smooth decay
        swoosh_env = np.exp(-4 * swoosh_t)
        swoosh = swoosh * swoosh_env

        # Combine parts
        diff_sound = np.zeros(samples, dtype=np.float32)
        diff_sound[:chunk_samples] = chunk.astype(np.float32)
        diff_sound[chunk_samples:] = swoosh.astype(np.float32)

        # Add a subtle "digital" beep undertone
        beep_freq = 1200
        beep = np.sin(2 * np.pi * beep_freq * t) * 0.1
        beep_env = np.exp(-8 * t)
        diff_sound = diff_sound + (beep * beep_env).astype(np.float32)

        # Normalize and apply volume
        diff_sound = self._normalize(diff_sound) * self.volume * 0.6

        return SoundEffect(
            audio=diff_sound.astype(np.float32),
            sample_rate=self.sample_rate,
            duration=duration,
            sfx_type=SFXType.CODE_DIFF,
        )


class SFXManager:
    """Manages sound effects for the commentary system."""

    def __init__(
        self,
        sample_rate: int = 44100,
        volume: float = 0.8,
        enabled: bool = True,
    ):
        """
        Initialize SFX manager.

        Args:
            sample_rate: Audio sample rate
            volume: SFX volume (0.0 to 1.0)
            enabled: Whether SFX are enabled
        """
        self.generator = SFXGenerator(sample_rate=sample_rate, volume=volume)
        self.enabled = enabled
        self.sample_rate = sample_rate
        self.volume = volume

    def get_sfx_for_event(self, event: str, mode: str = "sports") -> Optional[SoundEffect]:
        """
        Get appropriate sound effect for a detected event.

        Args:
            event: Event type string (e.g., "tests_pass", "error", "commit")
            mode: Commentary mode for mode-specific sounds

        Returns:
            SoundEffect or None if no SFX for this event
        """
        if not self.enabled:
            return None

        event_lower = event.lower()

        # Success events
        if any(
            x in event_lower
            for x in ["tests_pass", "test_pass", "all_tests", "build_success", "passed"]
        ):
            return self.generator.generate_air_horn()

        if any(x in event_lower for x in ["commit", "push", "merge", "deploy", "pr_created"]):
            return self.generator.generate_ding()

        if any(x in event_lower for x in ["milestone", "complete", "finished", "victory"]):
            return self.generator.generate_victory_fanfare()

        # Failure events
        if any(x in event_lower for x in ["tests_fail", "test_fail", "failed"]):
            return self.generator.generate_sad_trombone()

        if any(x in event_lower for x in ["error", "exception", "crash", "traceback"]):
            return self.generator.generate_glass_break()

        if any(x in event_lower for x in ["build_fail", "lint_fail", "type_error"]):
            return self.generator.generate_buzzer()

        # Big moment events (any mode)
        if any(x in event_lower for x in ["big_moment", "dramatic", "finisher", "major"]):
            return self.generator.generate_dramatic_hit()

        # Code diff/change events
        if any(x in event_lower for x in ["diff", "code_change", "refactor", "edit", "modify"]):
            return self.generator.generate_code_diff()

        # Start events
        if any(x in event_lower for x in ["start", "begin", "new_file", "created"]):
            if mode == "wwe":
                return self.generator.generate_bell_ring()
            else:
                return self.generator.generate_ding()

        return None

    def create_sfx_then_commentary(
        self,
        sfx: SoundEffect,
        commentary_audio: np.ndarray,
        commentary_sample_rate: int,
        gap_duration: float = 0.2,
    ) -> np.ndarray:
        """
        Create audio with SFX playing before commentary.

        Args:
            sfx: Sound effect to play first
            commentary_audio: Main commentary audio
            gap_duration: Pause between SFX and commentary (seconds)

        Returns:
            Combined audio array
        """
        sfx_audio = resample_audio(sfx.audio, sfx.sample_rate, commentary_sample_rate)
        gap_samples = int(gap_duration * commentary_sample_rate)
        gap = np.zeros(gap_samples, dtype=np.float32)

        return np.concatenate([sfx_audio, gap, commentary_audio.astype(np.float32)])
