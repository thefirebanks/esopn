"""Text-to-Speech module using Dia 1.6B for dual-speaker dialogue synthesis."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class SynthesizedAudio:
    """Container for synthesized audio data."""

    audio: np.ndarray  # Audio samples as numpy array
    sample_rate: int
    duration: float  # Duration in seconds
    dialogue: str  # Original dialogue text


class DiaTTS:
    """Text-to-Speech using Dia 1.6B model for natural dual-speaker dialogue."""

    def __init__(
        self,
        model_id: str = "nari-labs/Dia-1.6B-0626",
        device: Optional[Literal["cuda", "mps", "cpu"]] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize Dia TTS model.

        Args:
            model_id: HuggingFace model identifier
            device: Device to run inference on. Auto-detected if None.
            torch_dtype: Torch dtype for model. Auto-selected if None.
        """
        self.model_id = model_id
        self.device = device or self._detect_device()
        self.torch_dtype = torch_dtype or self._select_dtype()

        self._model = None
        self._processor = None
        self._loaded = False

    def _detect_device(self) -> str:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _select_dtype(self) -> torch.dtype:
        """Select appropriate dtype based on device."""
        if self.device == "cuda":
            return torch.bfloat16
        elif self.device == "mps":
            # Use float32 on MPS to avoid numerical instability issues
            # float16 on MPS can cause inf/nan in probability tensors
            return torch.float32
        return torch.float32

    def load(self) -> None:
        """Load the model and processor."""
        if self._loaded:
            return

        logger.info(f"Loading Dia TTS model: {self.model_id}")
        logger.info(f"Device: {self.device}, dtype: {self.torch_dtype}")

        try:
            from transformers import AutoProcessor, DiaForConditionalGeneration
            from rich.console import Console
            import os

            console = Console()
            
            # Check if model is already cached
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            model_cache = os.path.join(cache_dir, f"models--{self.model_id.replace('/', '--')}")
            
            # Check for incomplete downloads
            blobs_dir = os.path.join(model_cache, "blobs")
            incomplete_files = []
            total_incomplete_size = 0
            complete_size = 0
            if os.path.exists(blobs_dir):
                for f in os.listdir(blobs_dir):
                    fpath = os.path.join(blobs_dir, f)
                    size = os.path.getsize(fpath)
                    if f.endswith(".incomplete"):
                        incomplete_files.append((f, size))
                        total_incomplete_size += size
                    else:
                        complete_size += size
            
            total_downloaded = total_incomplete_size + complete_size
            is_first_download = not os.path.exists(model_cache) or incomplete_files
            
            if is_first_download:
                console.print(f"\n[bold cyan]📥 Downloading TTS model[/bold cyan] ({self.model_id})")
                console.print("   [dim]Model size: ~3.2GB. First run will take a few minutes.[/dim]")
                if total_downloaded > 0:
                    downloaded_gb = total_downloaded / (1024**3)
                    console.print(f"   [green]Resuming: {downloaded_gb:.1f}GB already downloaded[/green]")
                    if incomplete_files:
                        console.print(f"   [dim]({len(incomplete_files)} file(s) in progress)[/dim]")
                console.print()
            else:
                console.print(f"\n[bold green]📦 Loading TTS model from cache...[/bold green]")
            
            # Load processor first (small, fast)
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            
            # Load model (this is where the big download happens)
            # transformers will show its own progress bars here
            self._model = DiaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
            ).to(self.device)
            
            if is_first_download:
                console.print("\n[bold green]✓ Model downloaded and loaded![/bold green]")

            # Optional: compile for faster inference (PyTorch 2.0+)
            if hasattr(torch, "compile") and self.device == "cuda":
                logger.info("Compiling model with torch.compile...")
                self._model = torch.compile(self._model)

            self._loaded = True
            logger.info("Dia TTS model loaded successfully")

        except ImportError as e:
            raise ImportError(
                "Dia model requires transformers>=4.52.0. "
                "Install with: pip install transformers>=4.52.0"
            ) from e

    def synthesize(
        self,
        dialogue: str,
        max_new_tokens: int = 3072,
        guidance_scale: float = 3.0,
        temperature: float = 1.4,
    ) -> SynthesizedAudio:
        """
        Synthesize speech from dialogue text.

        Args:
            dialogue: Dialogue text with [S1] and [S2] speaker tags
            max_new_tokens: Maximum tokens to generate
            guidance_scale: CFG guidance scale (higher = more adherent to text)
            temperature: Sampling temperature (higher = more variation)

        Returns:
            SynthesizedAudio containing the generated audio
        """
        if not self._loaded:
            self.load()

        # Ensure dialogue has proper speaker tags
        if "[S1]" not in dialogue and "[S2]" not in dialogue:
            # Wrap in default speaker tag
            dialogue = f"[S1] {dialogue}"

        logger.debug(f"Synthesizing: {dialogue[:100]}...")

        # Prepare inputs
        inputs = self._processor(
            text=[dialogue],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate audio
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                guidance_scale=guidance_scale,
                temperature=temperature,
                do_sample=True,
            )

        # Decode audio
        audio_arrays = self._processor.batch_decode(outputs)

        if not audio_arrays or audio_arrays[0] is None:
            raise RuntimeError("Failed to generate audio")

        # Get the audio array (first item in batch)
        audio = audio_arrays[0]

        # Convert to numpy if tensor
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # Ensure float32 for audio playback compatibility
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize audio to [-1, 1] range
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        # Get sample rate from processor
        sample_rate = getattr(self._processor, "sampling_rate", 44100)

        duration = len(audio) / sample_rate

        return SynthesizedAudio(
            audio=audio,
            sample_rate=sample_rate,
            duration=duration,
            dialogue=dialogue,
        )

    def save_audio(
        self,
        audio: Union[SynthesizedAudio, np.ndarray],
        path: Union[str, Path],
        sample_rate: Optional[int] = None,
    ) -> None:
        """
        Save audio to file.

        Args:
            audio: SynthesizedAudio object or raw numpy array
            path: Output file path
            sample_rate: Sample rate (required if audio is numpy array)
        """
        import soundfile as sf

        if isinstance(audio, SynthesizedAudio):
            sf.write(path, audio.audio, audio.sample_rate)
        else:
            if sample_rate is None:
                raise ValueError("sample_rate required when audio is numpy array")
            sf.write(path, audio, sample_rate)

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        self._loaded = False

        # Clear CUDA cache if applicable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Dia TTS model unloaded")


class TTSManager:
    """Manager for TTS with provider selection and fallback options."""

    def __init__(
        self,
        provider: Literal["gemini", "elevenlabs", "dia"] = "gemini",
        # Gemini settings (default - FREE!)
        gemini_api_key: Optional[str] = None,
        gemini_alex_voice: str = "Fenrir",
        gemini_morgan_voice: str = "Kore",
        # ElevenLabs settings (fallback)
        elevenlabs_api_key: Optional[str] = None,
        elevenlabs_model: str = "eleven_turbo_v2_5",
        elevenlabs_alex_voice: Optional[str] = None,
        elevenlabs_morgan_voice: Optional[str] = None,
        # Dia settings (fallback for CUDA)
        dia_model_id: str = "nari-labs/Dia-1.6B-0626",
        dia_device: Optional[str] = None,
    ):
        """
        Initialize TTS manager.

        Args:
            provider: TTS provider to use ("gemini", "elevenlabs", or "dia")
            gemini_api_key: Gemini API key (same as vision API key)
            gemini_alex_voice: Gemini voice name for Alex
            gemini_morgan_voice: Gemini voice name for Morgan
            elevenlabs_api_key: ElevenLabs API key
            elevenlabs_model: ElevenLabs model ID
            elevenlabs_alex_voice: Voice ID for Alex
            elevenlabs_morgan_voice: Voice ID for Morgan
            dia_model_id: Dia model identifier
            dia_device: Device for Dia inference
        """
        self.provider = provider
        self._initialized = False
        self._tts = None
        self._sample_rate = 44100  # Default, updated based on provider

        # Store settings for lazy initialization
        self._gemini_settings = {
            "api_key": gemini_api_key,
            "alex_voice": gemini_alex_voice,
            "morgan_voice": gemini_morgan_voice,
        }
        self._elevenlabs_settings = {
            "api_key": elevenlabs_api_key,
            "model_id": elevenlabs_model,
            "alex_voice_id": elevenlabs_alex_voice,
            "morgan_voice_id": elevenlabs_morgan_voice,
        }
        self._dia_settings = {
            "model_id": dia_model_id,
            "device": dia_device,
        }

    @property
    def sample_rate(self) -> int:
        """Get the sample rate for the current provider."""
        return self._sample_rate

    def initialize(self) -> None:
        """Initialize TTS (load models/clients)."""
        if self._initialized:
            return

        if self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "elevenlabs":
            self._init_elevenlabs()
        else:
            self._init_dia()

        self._initialized = True

    def _init_gemini(self) -> None:
        """Initialize Gemini TTS."""
        from esopn.tts_gemini import GeminiTTS

        api_key = self._gemini_settings["api_key"]
        if not api_key:
            raise ValueError(
                "Gemini API key required. Set ESOPN_GEMINI_API_KEY environment variable."
            )

        logger.info("Initializing Gemini TTS (FREE multi-speaker)")
        self._tts = GeminiTTS(
            api_key=api_key,
            alex_voice=self._gemini_settings["alex_voice"],
            morgan_voice=self._gemini_settings["morgan_voice"],
        )
        self._sample_rate = 24000  # Gemini outputs at 24kHz

    def _init_elevenlabs(self) -> None:
        """Initialize ElevenLabs TTS."""
        from esopn.tts_elevenlabs import ElevenLabsTTS

        api_key = self._elevenlabs_settings["api_key"]
        if not api_key:
            raise ValueError(
                "ElevenLabs API key required. Set ESOPN_ELEVENLABS_API_KEY environment variable."
            )

        logger.info("Initializing ElevenLabs TTS")
        self._tts = ElevenLabsTTS(
            api_key=api_key,
            alex_voice_id=self._elevenlabs_settings["alex_voice_id"],
            morgan_voice_id=self._elevenlabs_settings["morgan_voice_id"],
            model_id=self._elevenlabs_settings["model_id"],
        )
        self._sample_rate = 44100

    def _init_dia(self) -> None:
        """Initialize Dia TTS."""
        logger.info("Initializing Dia TTS")
        device_literal: Optional[Literal["cuda", "mps", "cpu"]] = None
        device = self._dia_settings["device"]
        if device in ("cuda", "mps", "cpu"):
            device_literal = device  # type: ignore

        self._tts = DiaTTS(
            model_id=self._dia_settings["model_id"],
            device=device_literal,
        )
        self._tts.load()
        self._sample_rate = 44100

    def synthesize(self, dialogue: str, **kwargs) -> SynthesizedAudio:
        """
        Synthesize dialogue to speech.

        Args:
            dialogue: Dialogue text with [S1] and [S2] speaker tags
            **kwargs: Additional arguments passed to the TTS provider

        Returns:
            SynthesizedAudio containing the generated audio
        """
        if not self._initialized:
            self.initialize()

        if self.provider == "gemini":
            # Use Gemini (FREE!)
            from esopn.tts_gemini import GeminiTTS
            assert isinstance(self._tts, GeminiTTS)
            result = self._tts.synthesize_dialogue(dialogue)
            return SynthesizedAudio(
                audio=result.audio,
                sample_rate=result.sample_rate,
                duration=result.duration,
                dialogue=dialogue,
            )
        elif self.provider == "elevenlabs":
            # Use ElevenLabs
            from esopn.tts_elevenlabs import ElevenLabsTTS
            assert isinstance(self._tts, ElevenLabsTTS)
            result = self._tts.synthesize_dialogue(dialogue)
            return SynthesizedAudio(
                audio=result.audio,
                sample_rate=result.sample_rate,
                duration=result.duration,
                dialogue=dialogue,
            )
        else:
            # Use Dia
            assert isinstance(self._tts, DiaTTS)
            return self._tts.synthesize(dialogue, **kwargs)

    def get_usage(self) -> dict:
        """Get API usage stats (ElevenLabs only)."""
        if self.provider == "elevenlabs" and self._tts is not None:
            from esopn.tts_elevenlabs import ElevenLabsTTS
            if isinstance(self._tts, ElevenLabsTTS):
                return self._tts.get_character_usage()
        return {}

    def shutdown(self) -> None:
        """Shutdown and free resources."""
        if self.provider == "dia" and isinstance(self._tts, DiaTTS):
            self._tts.unload()
        self._tts = None
        self._initialized = False
