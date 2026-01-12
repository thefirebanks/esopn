"""ElevenLabs TTS provider for ESOPN commentary."""

import io
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Default voice IDs for the two commentators
# These are ElevenLabs stock voices - can be customized
DEFAULT_VOICES = {
    "alex": "pNInz6obpgDQGcFmaJgB",  # Adam - deep, authoritative (play-by-play)
    "morgan": "21m00Tcm4TlvDq8ikWAM",  # Rachel - warm, analytical (color commentary)
}


@dataclass
class ElevenLabsAudio:
    """Container for synthesized audio from ElevenLabs."""

    audio: np.ndarray
    sample_rate: int
    duration: float
    text: str
    voice_id: str


class ElevenLabsTTS:
    """ElevenLabs TTS client for dual-speaker commentary."""

    def __init__(
        self,
        api_key: str,
        alex_voice_id: Optional[str] = None,
        morgan_voice_id: Optional[str] = None,
        model_id: str = "eleven_turbo_v2_5",
    ):
        """
        Initialize ElevenLabs TTS.

        Args:
            api_key: ElevenLabs API key
            alex_voice_id: Voice ID for Alex (play-by-play). Uses default if None.
            morgan_voice_id: Voice ID for Morgan (color commentary). Uses default if None.
            model_id: ElevenLabs model to use. turbo_v2_5 is fast and good quality.
        """
        self.api_key = api_key
        self.alex_voice_id = alex_voice_id or DEFAULT_VOICES["alex"]
        self.morgan_voice_id = morgan_voice_id or DEFAULT_VOICES["morgan"]
        self.model_id = model_id
        self._client = None

    def _get_client(self):
        """Lazy-load ElevenLabs client."""
        if self._client is None:
            try:
                from elevenlabs import ElevenLabs
                self._client = ElevenLabs(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "ElevenLabs package not installed. "
                    "Install with: uv add elevenlabs"
                )
        return self._client

    def synthesize_single(
        self,
        text: str,
        voice_id: str,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
    ) -> ElevenLabsAudio:
        """
        Synthesize a single piece of text.

        Args:
            text: Text to synthesize
            voice_id: ElevenLabs voice ID
            stability: Voice stability (0-1). Lower = more expressive.
            similarity_boost: Voice similarity (0-1). Higher = more consistent.

        Returns:
            ElevenLabsAudio with synthesized audio
        """
        client = self._get_client()

        logger.debug(f"Synthesizing with voice {voice_id}: {text[:50]}...")

        # Generate audio
        audio_generator = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=self.model_id,
            voice_settings={
                "stability": stability,
                "similarity_boost": similarity_boost,
            },
            output_format="mp3_44100_128",
        )

        # Collect bytes from generator
        audio_bytes = b"".join(audio_generator)

        # Convert MP3 to numpy array
        audio, sample_rate = sf.read(io.BytesIO(audio_bytes))

        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Ensure float32
        audio = audio.astype(np.float32)

        duration = len(audio) / sample_rate

        return ElevenLabsAudio(
            audio=audio,
            sample_rate=sample_rate,
            duration=duration,
            text=text,
            voice_id=voice_id,
        )

    def synthesize_dialogue(
        self,
        dialogue: str,
        crossfade_ms: int = 100,
    ) -> ElevenLabsAudio:
        """
        Synthesize dialogue with [S1] and [S2] speaker tags.

        [S1] = Alex (play-by-play)
        [S2] = Morgan (color commentary)

        Args:
            dialogue: Text with [S1] and [S2] tags
            crossfade_ms: Milliseconds of crossfade between speakers

        Returns:
            ElevenLabsAudio with combined audio
        """
        # Parse dialogue into segments
        segments = self._parse_dialogue(dialogue)

        if not segments:
            raise ValueError("No valid dialogue segments found")

        # Synthesize each segment
        audio_segments = []
        sample_rate = None

        for speaker, text in segments:
            voice_id = self.alex_voice_id if speaker == "S1" else self.morgan_voice_id
            result = self.synthesize_single(text.strip(), voice_id)
            audio_segments.append(result.audio)
            sample_rate = result.sample_rate

        # Combine segments with crossfade
        combined = self._combine_with_crossfade(
            audio_segments, sample_rate, crossfade_ms
        )

        return ElevenLabsAudio(
            audio=combined,
            sample_rate=sample_rate,
            duration=len(combined) / sample_rate,
            text=dialogue,
            voice_id="mixed",
        )

    def _parse_dialogue(self, dialogue: str) -> list[tuple[str, str]]:
        """
        Parse dialogue text into (speaker, text) segments.

        Args:
            dialogue: Text with [S1] and [S2] tags

        Returns:
            List of (speaker, text) tuples
        """
        import re

        segments = []
        pattern = r"\[(S[12])\]\s*"

        # Split by speaker tags, keeping the tags
        parts = re.split(pattern, dialogue)

        # parts will be like: ['', 'S1', ' text1 ', 'S2', ' text2 ']
        i = 1  # Skip first empty part
        while i < len(parts) - 1:
            speaker = parts[i]
            text = parts[i + 1].strip()
            if text:
                segments.append((speaker, text))
            i += 2

        # If no tags found, treat entire text as S1
        if not segments and dialogue.strip():
            segments.append(("S1", dialogue.strip()))

        return segments

    def _combine_with_crossfade(
        self,
        segments: list[np.ndarray],
        sample_rate: int,
        crossfade_ms: int,
    ) -> np.ndarray:
        """
        Combine audio segments with crossfade transitions.

        Args:
            segments: List of audio arrays
            sample_rate: Audio sample rate
            crossfade_ms: Crossfade duration in milliseconds

        Returns:
            Combined audio array
        """
        if len(segments) == 1:
            return segments[0]

        crossfade_samples = int(sample_rate * crossfade_ms / 1000)

        # Calculate total length
        total_length = sum(len(s) for s in segments)
        total_length -= crossfade_samples * (len(segments) - 1)

        combined = np.zeros(total_length, dtype=np.float32)
        pos = 0

        for i, segment in enumerate(segments):
            if i == 0:
                # First segment: copy entirely
                combined[:len(segment)] = segment
                pos = len(segment) - crossfade_samples
            else:
                # Subsequent segments: crossfade
                fade_in = np.linspace(0, 1, crossfade_samples, dtype=np.float32)
                fade_out = np.linspace(1, 0, crossfade_samples, dtype=np.float32)

                # Apply crossfade to overlap region
                combined[pos:pos + crossfade_samples] *= fade_out
                combined[pos:pos + crossfade_samples] += segment[:crossfade_samples] * fade_in

                # Copy rest of segment
                remaining = segment[crossfade_samples:]
                combined[pos + crossfade_samples:pos + crossfade_samples + len(remaining)] = remaining

                pos = pos + len(segment) - crossfade_samples

        return combined

    def get_character_usage(self) -> dict:
        """
        Get current API usage stats.

        Returns:
            Dict with character usage info
        """
        client = self._get_client()
        try:
            user = client.user.get()
            subscription = user.subscription
            return {
                "character_count": subscription.character_count,
                "character_limit": subscription.character_limit,
                "remaining": subscription.character_limit - subscription.character_count,
            }
        except Exception as e:
            logger.warning(f"Failed to get usage stats: {e}")
            return {}

    def list_voices(self) -> list[dict]:
        """
        List available voices.

        Returns:
            List of voice info dicts
        """
        client = self._get_client()
        voices = client.voices.get_all()
        return [
            {
                "voice_id": v.voice_id,
                "name": v.name,
                "category": getattr(v, "category", "unknown"),
            }
            for v in voices.voices
        ]
