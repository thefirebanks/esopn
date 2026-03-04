"""Audio playback module for real-time audio output."""

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .audio_utils import resample_audio

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """A chunk of audio to be played."""

    audio: np.ndarray
    sample_rate: int
    priority: int = 0  # Higher priority plays first


class AudioPlayer:
    """Real-time audio player using sounddevice."""

    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 1,
        buffer_size: int = 10,
    ):
        """
        Initialize the audio player.

        Args:
            sample_rate: Default sample rate for playback
            channels: Number of audio channels (1=mono, 2=stereo)
            buffer_size: Maximum number of audio chunks to buffer
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self._queue: queue.PriorityQueue[tuple[int, int, AudioChunk]] = queue.PriorityQueue(
            maxsize=buffer_size
        )
        self._playing = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._chunk_counter = 0
        self._chunk_counter_lock = threading.Lock()
        self._currently_playing = threading.Event()
        self._playback_lock = threading.Lock()

    def start(self) -> None:
        """Start the audio playback thread."""
        if self._playing:
            return

        self._stop_event.clear()
        self._playing = True
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()
        logger.info("Audio player started")

    def stop(self) -> None:
        """Stop the audio playback thread."""
        if not self._playing:
            return

        self._stop_event.set()
        self._playing = False

        import sounddevice as sd

        with self._playback_lock:
            sd.stop()

        # Clear the queue
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        logger.info("Audio player stopped")

    def play(self, audio: np.ndarray, sample_rate: Optional[int] = None, priority: int = 0) -> None:
        """
        Queue audio for playback.

        Args:
            audio: Audio samples as numpy array
            sample_rate: Sample rate (uses default if None)
            priority: Playback priority (higher = plays sooner)
        """
        if not self._playing:
            self.start()

        sr = sample_rate or self.sample_rate
        chunk = AudioChunk(audio=audio, sample_rate=sr, priority=priority)

        # Priority queue: lower number = higher priority
        # Use negative priority so higher priority values play first
        # Add counter to maintain FIFO order for same priority
        with self._chunk_counter_lock:
            self._chunk_counter += 1
            counter = self._chunk_counter
        try:
            self._queue.put((-priority, counter, chunk), block=False)
        except queue.Full:
            logger.warning("Audio queue full, dropping chunk")

    def play_sync(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> None:
        """
        Play audio synchronously (blocking).

        Args:
            audio: Audio samples as numpy array
            sample_rate: Sample rate (uses default if None)
        """
        import sounddevice as sd

        sr = sample_rate or self.sample_rate

        # Ensure audio is the right shape
        if audio.ndim == 1 and self.channels == 1:
            pass  # Already correct
        elif audio.ndim == 1 and self.channels == 2:
            audio = np.column_stack([audio, audio])

        with self._playback_lock:
            self._currently_playing.set()
            try:
                sd.play(audio, sr)
                sd.wait()
            finally:
                self._currently_playing.clear()

    def _playback_loop(self) -> None:
        """Background thread for audio playback."""
        import sounddevice as sd

        while not self._stop_event.is_set():
            try:
                # Get next chunk with timeout
                _, _, chunk = self._queue.get(timeout=0.1)

                # Resample if needed
                audio = chunk.audio
                if chunk.sample_rate != self.sample_rate:
                    audio = self._resample(audio, chunk.sample_rate, self.sample_rate)

                # Ensure correct shape
                if audio.ndim == 1 and self.channels == 1:
                    pass
                elif audio.ndim == 1 and self.channels == 2:
                    audio = np.column_stack([audio, audio])

                # Play audio
                with self._playback_lock:
                    self._currently_playing.set()
                    try:
                        sd.play(audio, self.sample_rate)
                        sd.wait()
                    finally:
                        self._currently_playing.clear()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio playback error: {e}")

    def _resample(self, audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """Simple resampling using linear interpolation."""
        return resample_audio(audio, src_rate, dst_rate)

    @property
    def is_playing(self) -> bool:
        """Check if audio is currently being played."""
        return self._playing and (not self._queue.empty() or self._currently_playing.is_set())

    def wait(self) -> None:
        """Wait for all queued audio to finish playing."""
        while not self._queue.empty() or self._currently_playing.is_set():
            import time

            time.sleep(0.1)


class AudioManager:
    """High-level audio management for the commentator system."""

    def __init__(self, sample_rate: int = 44100):
        """Initialize audio manager."""
        self.player = AudioPlayer(sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self._started = False

        # Ambient sound support
        self._ambient_thread: Optional[threading.Thread] = None
        self._ambient_stop_event = threading.Event()
        self._ambient_pause_event = threading.Event()  # Pause ambient during commentary
        self._ambient_audio: Optional[np.ndarray] = None
        self._ambient_playing = False
        self._ambient_volume = 0.12

    def start(self) -> None:
        """Start the audio system."""
        if self._started:
            return

        self.player.start()
        self._started = True

    def set_ambient(self, audio: np.ndarray, volume: float = 0.12) -> None:
        """
        Set the ambient background audio to loop.

        Args:
            audio: Ambient audio samples (will be looped)
            volume: Ambient volume (0.0 to 1.0)
        """
        self._ambient_audio = audio * volume
        self._ambient_volume = volume

    def start_ambient(self) -> None:
        """Start playing ambient audio in a loop."""
        if self._ambient_audio is None:
            logger.warning("No ambient audio set")
            return

        if self._ambient_playing:
            return

        self._ambient_stop_event.clear()
        self._ambient_playing = True
        self._ambient_thread = threading.Thread(target=self._ambient_loop, daemon=True)
        self._ambient_thread.start()
        logger.info("Ambient audio started")

    def stop_ambient(self) -> None:
        """Stop the ambient audio loop."""
        if not self._ambient_playing:
            return

        self._ambient_stop_event.set()
        self._ambient_playing = False

        if self._ambient_thread:
            self._ambient_thread.join(timeout=2.0)
            self._ambient_thread = None

        logger.info("Ambient audio stopped")

    def _ambient_loop(self) -> None:
        """Background thread for looping ambient audio."""
        if self._ambient_audio is None:
            return

        while not self._ambient_stop_event.is_set():
            try:
                # Wait if paused (during commentary playback)
                if self._ambient_pause_event.is_set():
                    import time

                    time.sleep(0.1)
                    continue

                duration = len(self._ambient_audio) / self.sample_rate
                self.player.play(self._ambient_audio, self.sample_rate, priority=-10)

                import time

                end_time = time.time() + duration
                while time.time() < end_time:
                    if self._ambient_stop_event.is_set() or self._ambient_pause_event.is_set():
                        break
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"Ambient playback error: {e}")
                break

    def play_commentary(self, audio: np.ndarray, sample_rate: int) -> None:
        """Play commentary audio."""
        if not self._started:
            self.start()

        self.player.play(audio, sample_rate, priority=10)

    def play_commentary_sync(self, audio: np.ndarray, sample_rate: int) -> None:
        """Play commentary audio synchronously (pauses ambient during playback)."""
        # Pause ambient while commentary plays
        self._ambient_pause_event.set()
        self.player.stop()

        # Play commentary
        self.player.play_sync(audio, sample_rate)

        self.player.start()

        # Resume ambient
        self._ambient_pause_event.clear()

    def stop_all(self) -> None:
        """Stop all audio playback immediately (for pause functionality)."""
        self.player.stop()
        self._ambient_pause_event.set()

    def shutdown(self) -> None:
        """Shutdown the audio system."""
        self.stop_ambient()
        self.player.stop()
        self._started = False
