"""Crowd sounds module - generates synthetic crowd reactions based on intensity."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class CrowdReaction(Enum):
    """Types of crowd reactions."""
    
    SILENCE = "silence"           # Intensity 1-2: Quiet, focused coding
    MURMUR = "murmur"             # Intensity 3-4: Light background chatter
    INTERESTED = "interested"     # Intensity 5-6: Engaged "ooh" sounds
    EXCITED = "excited"           # Intensity 7-8: Building excitement
    CHEERING = "cheering"         # Intensity 9-10: Full crowd roar
    GASP = "gasp"                 # Special: Error or unexpected event
    APPLAUSE = "applause"         # Special: Success, build passing


@dataclass
class CrowdAudio:
    """Container for crowd audio."""
    
    audio: np.ndarray
    sample_rate: int
    duration: float
    reaction: CrowdReaction


class CrowdSoundGenerator:
    """Generates synthetic crowd sounds using signal processing."""
    
    def __init__(self, sample_rate: int = 44100, volume: float = 0.3):
        """
        Initialize crowd sound generator.
        
        Args:
            sample_rate: Audio sample rate
            volume: Base volume level (0.0 to 1.0)
        """
        self.sample_rate = sample_rate
        self.volume = volume
        self._rng = np.random.default_rng()
    
    def _pink_noise(self, duration: float, amplitude: float = 1.0) -> np.ndarray:
        """Generate pink noise (1/f noise) - sounds more natural than white noise."""
        samples = int(duration * self.sample_rate)
        
        # Generate white noise
        white = self._rng.standard_normal(samples)
        
        # Apply 1/f filter using cumulative sum method (simple approximation)
        # This creates a more natural, "crowd-like" rumble
        pink = np.zeros(samples)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        
        # Simple IIR filter for pink noise approximation
        for i in range(len(b), samples):
            pink[i] = (b[0] * white[i] + b[1] * white[i-1] + 
                      b[2] * white[i-2] + b[3] * white[i-3] -
                      a[1] * pink[i-1] - a[2] * pink[i-2] - a[3] * pink[i-3])
        
        # Normalize and apply amplitude
        pink = pink / (np.abs(pink).max() + 1e-8) * amplitude
        return pink.astype(np.float32)
    
    def _apply_envelope(
        self, 
        audio: np.ndarray, 
        attack: float = 0.1,
        decay: float = 0.1,
        sustain: float = 0.8,
        release: float = 0.2,
    ) -> np.ndarray:
        """Apply ADSR envelope to audio."""
        samples = len(audio)
        envelope = np.ones(samples, dtype=np.float32)
        
        attack_samples = int(attack * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        
        # Attack
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Release
        if release_samples > 0:
            envelope[-release_samples:] = np.linspace(1, 0, release_samples) * sustain
        
        # Sustain level for middle section
        if attack_samples < samples - release_samples:
            envelope[attack_samples:-release_samples] = sustain
        
        return audio * envelope
    
    def _modulate_amplitude(
        self, 
        audio: np.ndarray, 
        freq: float = 2.0,
        depth: float = 0.3,
    ) -> np.ndarray:
        """Add amplitude modulation for more organic sound."""
        t = np.linspace(0, len(audio) / self.sample_rate, len(audio))
        modulation = 1 + depth * np.sin(2 * np.pi * freq * t + self._rng.random() * 2 * np.pi)
        return audio * modulation.astype(np.float32)
    
    def _add_voices(
        self,
        duration: float,
        num_voices: int = 5,
        base_freq: float = 200,
        freq_spread: float = 100,
    ) -> np.ndarray:
        """Add synthetic voice-like tones to simulate crowd vocals."""
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        voices = np.zeros(samples, dtype=np.float32)
        
        for _ in range(num_voices):
            # Random frequency for each "voice"
            freq = base_freq + self._rng.random() * freq_spread
            # Add some vibrato
            vibrato = 5 * np.sin(2 * np.pi * 5 * t + self._rng.random() * 2 * np.pi)
            # Generate tone with vibrato
            voice = np.sin(2 * np.pi * (freq + vibrato) * t)
            # Random amplitude envelope
            amp_mod = self._rng.random() * 0.3 + 0.2
            voices += voice * amp_mod
        
        # Normalize
        voices = voices / (np.abs(voices).max() + 1e-8) * 0.3
        return voices.astype(np.float32)
    
    def generate_silence(self, duration: float = 0.5) -> CrowdAudio:
        """Generate silence (or very quiet ambient)."""
        samples = int(duration * self.sample_rate)
        # Just a tiny bit of room tone
        audio = self._pink_noise(duration, amplitude=0.02) * self.volume
        
        return CrowdAudio(
            audio=audio.astype(np.float32),
            sample_rate=self.sample_rate,
            duration=duration,
            reaction=CrowdReaction.SILENCE,
        )
    
    def generate_murmur(self, duration: float = 2.0) -> CrowdAudio:
        """Generate quiet crowd murmur."""
        # Low rumble with gentle modulation
        base = self._pink_noise(duration, amplitude=0.15)
        base = self._modulate_amplitude(base, freq=0.5, depth=0.2)
        base = self._apply_envelope(base, attack=0.3, release=0.5, sustain=0.7)
        
        audio = base * self.volume
        
        return CrowdAudio(
            audio=audio.astype(np.float32),
            sample_rate=self.sample_rate,
            duration=duration,
            reaction=CrowdReaction.MURMUR,
        )
    
    def generate_interested(self, duration: float = 1.5) -> CrowdAudio:
        """Generate interested crowd sound - 'ooh' reaction."""
        # Pink noise base
        base = self._pink_noise(duration, amplitude=0.25)
        
        # Add voice-like tones (lower, curious sound)
        voices = self._add_voices(duration, num_voices=8, base_freq=180, freq_spread=60)
        
        # Combine
        audio = base * 0.4 + voices * 0.6
        audio = self._modulate_amplitude(audio, freq=1.0, depth=0.25)
        audio = self._apply_envelope(audio, attack=0.2, release=0.4, sustain=0.8)
        
        audio = audio * self.volume
        
        return CrowdAudio(
            audio=audio.astype(np.float32),
            sample_rate=self.sample_rate,
            duration=duration,
            reaction=CrowdReaction.INTERESTED,
        )
    
    def generate_excited(self, duration: float = 2.0) -> CrowdAudio:
        """Generate excited crowd - building energy."""
        # Louder pink noise
        base = self._pink_noise(duration, amplitude=0.4)
        
        # More voices, higher pitch
        voices = self._add_voices(duration, num_voices=15, base_freq=250, freq_spread=150)
        
        # Combine with rising intensity
        audio = base * 0.3 + voices * 0.7
        
        # Rising envelope
        samples = len(audio)
        rise = np.linspace(0.6, 1.0, samples).astype(np.float32)
        audio = audio * rise
        
        audio = self._modulate_amplitude(audio, freq=2.0, depth=0.3)
        audio = self._apply_envelope(audio, attack=0.15, release=0.3, sustain=0.9)
        
        audio = audio * self.volume
        
        return CrowdAudio(
            audio=audio.astype(np.float32),
            sample_rate=self.sample_rate,
            duration=duration,
            reaction=CrowdReaction.EXCITED,
        )
    
    def generate_cheering(self, duration: float = 3.0) -> CrowdAudio:
        """Generate full crowd cheering - maximum energy!"""
        # Loud pink noise base
        base = self._pink_noise(duration, amplitude=0.5)
        
        # Lots of voices, high energy
        voices = self._add_voices(duration, num_voices=25, base_freq=300, freq_spread=200)
        
        # Add some high frequency excitement
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        excitement = np.sin(2 * np.pi * 800 * t) * 0.1 * self._rng.random(samples)
        
        # Combine
        audio = base * 0.25 + voices * 0.65 + excitement.astype(np.float32) * 0.1
        
        # Strong modulation for wave-like effect
        audio = self._modulate_amplitude(audio, freq=1.5, depth=0.35)
        audio = self._apply_envelope(audio, attack=0.1, release=0.5, sustain=1.0)
        
        audio = audio * self.volume * 1.2  # Slightly louder
        
        return CrowdAudio(
            audio=audio.astype(np.float32),
            sample_rate=self.sample_rate,
            duration=duration,
            reaction=CrowdReaction.CHEERING,
        )
    
    def generate_gasp(self, duration: float = 1.0) -> CrowdAudio:
        """Generate crowd gasp - sudden intake of breath."""
        samples = int(duration * self.sample_rate)
        
        # Sharp attack, quick decay
        base = self._pink_noise(duration, amplitude=0.35)
        
        # High-pitched intake sound
        t = np.linspace(0, duration, samples)
        intake = np.sin(2 * np.pi * 400 * t) * np.exp(-3 * t)
        
        audio = base * 0.5 + intake.astype(np.float32) * 0.5
        
        # Sharp envelope
        audio = self._apply_envelope(audio, attack=0.05, release=0.3, sustain=0.5)
        
        audio = audio * self.volume
        
        return CrowdAudio(
            audio=audio.astype(np.float32),
            sample_rate=self.sample_rate,
            duration=duration,
            reaction=CrowdReaction.GASP,
        )
    
    def generate_applause(self, duration: float = 3.0) -> CrowdAudio:
        """Generate applause - clapping sounds."""
        samples = int(duration * self.sample_rate)
        
        # Base is filtered noise that sounds like many hands clapping
        audio = np.zeros(samples, dtype=np.float32)
        
        # Generate many individual "claps"
        clap_rate = 4  # claps per second per "person"
        num_clappers = 30
        
        for _ in range(num_clappers):
            # Random phase offset for this clapper
            offset = self._rng.random() * self.sample_rate // clap_rate
            
            # Slightly randomized clap timing
            for clap_time in np.arange(offset, samples, self.sample_rate // clap_rate):
                clap_time = int(clap_time + self._rng.integers(-500, 500))
                if 0 <= clap_time < samples - 1000:
                    # Short burst of noise for each clap
                    clap_duration = int(0.02 * self.sample_rate)  # 20ms
                    clap = self._rng.standard_normal(clap_duration) * 0.1
                    # Sharp envelope
                    clap_env = np.exp(-np.linspace(0, 5, clap_duration))
                    clap = clap * clap_env
                    
                    # Add to audio
                    end_idx = min(clap_time + clap_duration, samples)
                    audio[clap_time:end_idx] += clap[:end_idx - clap_time].astype(np.float32)
        
        # Add some pink noise underneath
        base = self._pink_noise(duration, amplitude=0.15)
        audio = audio + base
        
        # Normalize
        audio = audio / (np.abs(audio).max() + 1e-8)
        
        # Envelope
        audio = self._apply_envelope(audio, attack=0.2, release=0.5, sustain=0.8)
        
        audio = audio * self.volume
        
        return CrowdAudio(
            audio=audio.astype(np.float32),
            sample_rate=self.sample_rate,
            duration=duration,
            reaction=CrowdReaction.APPLAUSE,
        )
    
    def get_reaction_for_intensity(self, intensity: int) -> CrowdReaction:
        """Map intensity (1-10) to appropriate crowd reaction."""
        if intensity <= 2:
            return CrowdReaction.SILENCE
        elif intensity <= 4:
            return CrowdReaction.MURMUR
        elif intensity <= 6:
            return CrowdReaction.INTERESTED
        elif intensity <= 8:
            return CrowdReaction.EXCITED
        else:
            return CrowdReaction.CHEERING
    
    def generate_for_intensity(
        self, 
        intensity: int, 
        duration: Optional[float] = None,
        mood: Optional[str] = None,
    ) -> CrowdAudio:
        """
        Generate crowd sound based on intensity level.
        
        Args:
            intensity: Intensity level 1-10
            duration: Optional duration override
            mood: Optional mood hint ("error", "success", etc.)
            
        Returns:
            CrowdAudio for the appropriate reaction
        """
        # Check for special moods
        if mood:
            mood_lower = mood.lower()
            if "error" in mood_lower or "fail" in mood_lower or "bug" in mood_lower:
                return self.generate_gasp(duration or 1.0)
            elif "success" in mood_lower or "pass" in mood_lower or "complete" in mood_lower:
                return self.generate_applause(duration or 2.5)
        
        # Map intensity to reaction
        reaction = self.get_reaction_for_intensity(intensity)
        
        generators = {
            CrowdReaction.SILENCE: lambda: self.generate_silence(duration or 0.5),
            CrowdReaction.MURMUR: lambda: self.generate_murmur(duration or 2.0),
            CrowdReaction.INTERESTED: lambda: self.generate_interested(duration or 1.5),
            CrowdReaction.EXCITED: lambda: self.generate_excited(duration or 2.0),
            CrowdReaction.CHEERING: lambda: self.generate_cheering(duration or 3.0),
        }
        
        return generators[reaction]()


class CrowdManager:
    """Manages crowd sounds for the commentary system."""
    
    def __init__(
        self, 
        sample_rate: int = 44100,
        volume: float = 0.25,
        enabled: bool = True,
        ambient_enabled: bool = True,
        ambient_volume: float = 0.12,
    ):
        """
        Initialize crowd manager.
        
        Args:
            sample_rate: Audio sample rate
            volume: Crowd reaction volume (0.0 to 1.0)
            enabled: Whether crowd reaction sounds are enabled
            ambient_enabled: Whether continuous ambient crowd is enabled
            ambient_volume: Ambient crowd volume (0.0 to 1.0)
        """
        self.generator = CrowdSoundGenerator(sample_rate=sample_rate, volume=volume)
        self.ambient_generator = CrowdSoundGenerator(sample_rate=sample_rate, volume=ambient_volume)
        self.enabled = enabled
        self.ambient_enabled = ambient_enabled
        self.sample_rate = sample_rate
        self._last_intensity = 0
    
    def get_crowd_audio(
        self,
        intensity: int,
        mood: Optional[str] = None,
        duration: Optional[float] = None,
    ) -> Optional[CrowdAudio]:
        """
        Get crowd audio for current scene.
        
        Args:
            intensity: Scene intensity (1-10)
            mood: Scene mood string
            duration: Optional duration override
            
        Returns:
            CrowdAudio or None if disabled
        """
        if not self.enabled:
            return None
        
        self._last_intensity = intensity
        return self.generator.generate_for_intensity(intensity, duration, mood)
    
    def mix_with_commentary(
        self,
        commentary_audio: np.ndarray,
        crowd_audio: CrowdAudio,
        crowd_position: str = "under",  # "before", "under", "after"
    ) -> np.ndarray:
        """
        Mix crowd audio with commentary.
        
        Args:
            commentary_audio: Main commentary audio
            crowd_audio: Crowd reaction audio
            crowd_position: Where to place crowd sound
            
        Returns:
            Mixed audio array
        """
        if crowd_position == "before":
            # Crowd plays, then commentary
            padding = np.zeros(int(0.2 * self.sample_rate), dtype=np.float32)
            return np.concatenate([crowd_audio.audio, padding, commentary_audio])
        
        elif crowd_position == "after":
            # Commentary plays, then crowd
            padding = np.zeros(int(0.2 * self.sample_rate), dtype=np.float32)
            return np.concatenate([commentary_audio, padding, crowd_audio.audio])
        
        else:  # "under" - mix underneath commentary
            # Ensure same length
            commentary_len = len(commentary_audio)
            crowd_len = len(crowd_audio.audio)
            
            if crowd_len < commentary_len:
                # Pad crowd audio
                padded_crowd = np.zeros(commentary_len, dtype=np.float32)
                padded_crowd[:crowd_len] = crowd_audio.audio
                crowd_audio_matched = padded_crowd
            else:
                # Trim crowd audio
                crowd_audio_matched = crowd_audio.audio[:commentary_len]
            
            # Mix with commentary louder than crowd
            mixed = commentary_audio * 0.85 + crowd_audio_matched * 0.15
            
            # Normalize to prevent clipping
            max_val = np.abs(mixed).max()
            if max_val > 1.0:
                mixed = mixed / max_val
            
            return mixed
    
    def generate_ambient(self, duration: float) -> Optional[CrowdAudio]:
        """
        Generate continuous ambient crowd sound for filling silence.
        
        This is a gentle, low-key crowd murmur that plays between commentary
        to maintain the stadium atmosphere.
        
        Args:
            duration: Duration in seconds
            
        Returns:
            CrowdAudio with ambient sound, or None if disabled
        """
        if not self.ambient_enabled:
            return None
        
        return self.ambient_generator.generate_murmur(duration=duration)
    
    def generate_ambient_loop(self, duration: float) -> Optional[np.ndarray]:
        """
        Generate a loopable ambient crowd sound.
        
        The audio fades in/out at edges for seamless looping.
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Numpy array of ambient audio, or None if disabled
        """
        if not self.ambient_enabled:
            return None
        
        # Generate base ambient
        ambient = self.ambient_generator.generate_murmur(duration=duration)
        audio = ambient.audio
        
        # Apply fade in/out for smooth looping
        fade_samples = int(0.5 * self.sample_rate)  # 0.5s fade
        if len(audio) > fade_samples * 2:
            fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
            fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
            audio[:fade_samples] *= fade_in
            audio[-fade_samples:] *= fade_out
        
        return audio
