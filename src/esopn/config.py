"""Configuration management for ESOPN."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .personas import get_available_modes


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ESOPN_",
        extra="ignore",
    )

    # Screenshot settings
    capture_interval: float = Field(
        default=3.0,
        ge=0.5,
        le=30.0,
        description="Seconds between screenshots",
    )
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
    elevenlabs_model: str = Field(default="eleven_turbo_v2_5", description="ElevenLabs model ID")
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
    audio_sample_rate: int = Field(
        default=44100,
        ge=8000,
        le=96000,
        description="Audio sample rate",
    )

    # Commentary settings
    commentary_style: Literal["excited", "analytical", "balanced"] = Field(
        default="excited", description="Overall commentary style"
    )
    commentary_mode: str = Field(default="sports", description="Commentary mode/persona style")

    # Crowd sounds settings
    crowd_enabled: bool = Field(default=True, description="Enable crowd background sounds")
    crowd_volume: float = Field(default=0.25, ge=0.0, le=1.0, description="Crowd sound volume")
    crowd_ambient_enabled: bool = Field(
        default=False, description="Enable continuous ambient crowd sounds between commentary"
    )
    crowd_ambient_volume: float = Field(
        default=0.12,
        ge=0.0,
        le=1.0,
        description="Ambient crowd volume",
    )

    # Sound effects settings
    sfx_enabled: bool = Field(
        default=True, description="Enable reactive sound effects (air horn, sad trombone, etc.)"
    )
    sfx_volume: float = Field(default=0.8, ge=0.0, le=1.0, description="Sound effects volume")

    # Paths
    cache_dir: Path = Field(default=Path.home() / ".cache" / "esopn", description="Cache directory")

    @field_validator("commentary_mode")
    @classmethod
    def validate_commentary_mode(cls, value: str) -> str:
        """Validate commentary mode against registered personas."""
        valid_modes = set(get_available_modes())
        if value not in valid_modes:
            modes = ", ".join(sorted(valid_modes))
            raise ValueError(f"Invalid commentary mode '{value}'. Valid modes: {modes}")
        return value


def get_settings() -> Settings:
    """Get application settings."""
    settings = Settings()
    os.makedirs(settings.cache_dir, mode=0o700, exist_ok=True)
    return settings


@lru_cache(maxsize=1)
def get_cached_settings() -> Settings:
    """Get cached application settings for consistent runtime values."""
    return get_settings()
