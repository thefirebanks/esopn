"""Configuration management for ESOPN."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ESOPN_",
        extra="ignore",
    )

    # Screenshot settings
    capture_interval: float = Field(default=3.0, description="Seconds between screenshots")
    capture_monitor: int = Field(default=1, description="Monitor index to capture (1-based)")

    # Vision model settings
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    vision_model: str = Field(default="gemini-2.0-flash", description="Vision model to use")

    # TTS Provider settings
    tts_provider: Literal["gemini", "elevenlabs", "dia"] = Field(
        default="gemini", description="TTS provider to use (gemini, elevenlabs, or dia)"
    )

    # Gemini TTS settings (default - FREE!)
    gemini_alex_voice: str = Field(
        default="Fenrir",  # Excitable - perfect for play-by-play
        description="Gemini voice name for Alex (play-by-play)",
    )
    gemini_morgan_voice: str = Field(
        default="Kore",  # Firm - good for analytical commentary
        description="Gemini voice name for Morgan (color commentary)",
    )

    # ElevenLabs TTS settings (fallback - uses credits)
    elevenlabs_api_key: str = Field(default="", description="ElevenLabs API key")
    elevenlabs_model: str = Field(
        default="eleven_turbo_v2_5", description="ElevenLabs model ID"
    )
    elevenlabs_alex_voice: str = Field(
        default="pNInz6obpgDQGcFmaJgB",  # Adam - deep, authoritative
        description="ElevenLabs voice ID for Alex (play-by-play)",
    )
    elevenlabs_morgan_voice: str = Field(
        default="21m00Tcm4TlvDq8ikWAM",  # Rachel - warm, analytical
        description="ElevenLabs voice ID for Morgan (color commentary)",
    )

    # Dia TTS settings (fallback for CUDA users)
    tts_model: str = Field(
        default="nari-labs/Dia-1.6B-0626", description="Dia TTS model identifier"
    )
    tts_device: Optional[str] = Field(
        default=None, description="Device for TTS inference (cuda, mps, cpu, or None for auto)"
    )

    # Audio settings
    audio_sample_rate: int = Field(default=44100, description="Audio sample rate")

    # Commentary settings
    commentary_style: Literal["excited", "analytical", "balanced"] = Field(
        default="excited", description="Overall commentary style"
    )

    # Crowd sounds settings
    crowd_enabled: bool = Field(
        default=True, description="Enable crowd background sounds"
    )
    crowd_volume: float = Field(
        default=0.25, description="Crowd sound volume (0.0 to 1.0)"
    )
    crowd_ambient_enabled: bool = Field(
        default=False, description="Enable continuous ambient crowd sounds between commentary"
    )
    crowd_ambient_volume: float = Field(
        default=0.12, description="Ambient crowd volume (0.0 to 1.0)"
    )

    # Paths
    cache_dir: Path = Field(default=Path.home() / ".cache" / "esopn", description="Cache directory")


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
