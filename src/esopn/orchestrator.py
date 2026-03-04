"""Main orchestrator that coordinates all components for real-time commentary."""

import logging
import queue
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from multiprocessing import Queue
from typing import Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .audio import AudioManager
from .capture import ScreenCapture, Screenshot
from .commentary import Commentary, CommentaryGenerator, get_fallback_commentary
from .config import Settings
from .control import Command
from .crowd import CrowdManager
from .personas import get_voices
from .sfx import SFXManager
from .tts import SynthesizedAudio, TTSManager
from .vision import SceneAnalysis, VisionAnalyzer

logger = logging.getLogger(__name__)
console = Console()

# Default hotkey: Ctrl+Shift+P (P for Pause)
DEFAULT_HOTKEY = "<ctrl>+<shift>+p"


@dataclass
class CommentaryState:
    """Tracks the state of the commentary session."""

    screenshots_captured: int = 0
    analyses_completed: int = 0
    commentaries_generated: int = 0
    commentaries_skipped: int = 0  # Skipped due to no change
    audio_played: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    last_scene: Optional[SceneAnalysis] = None
    last_commentary: Optional[Commentary] = None
    last_screenshot: Optional[Screenshot] = None  # For change detection
    recent_scenes: deque[SceneAnalysis] = field(default_factory=lambda: deque(maxlen=5))
    paused: bool = False

    @property
    def uptime(self) -> float:
        """Get session uptime in seconds."""
        return time.time() - self.start_time


class HotkeyListener:
    """Listens for hotkey presses to toggle pause/resume."""

    def __init__(self, hotkey: str, callback: Callable[[], None]):
        """
        Initialize hotkey listener.

        Args:
            hotkey: Hotkey string (e.g., "<ctrl>+<shift>+p")
            callback: Function to call when hotkey is pressed
        """
        self.hotkey = hotkey
        self.callback = callback
        self._listener = None
        self._running = False

    def start(self) -> bool:
        """Start listening for hotkeys. Returns True if successful."""
        try:
            from pynput import keyboard

            def on_activate():
                self.callback()

            self._listener = keyboard.GlobalHotKeys({self.hotkey: on_activate})
            self._listener.start()
            self._running = True
            return True

        except ImportError:
            logger.warning("pynput not available, hotkey support disabled")
            return False
        except Exception as e:
            logger.warning(f"Could not start hotkey listener: {e}")
            return False

    def stop(self) -> None:
        """Stop listening for hotkeys."""
        if self._listener:
            self._listener.stop()
            if hasattr(self._listener, "join"):
                self._listener.join(timeout=1.0)
            self._listener = None
        self._running = False


class Orchestrator:
    """Coordinates screenshot capture, analysis, commentary generation, and TTS."""

    def __init__(
        self,
        settings: Settings,
        active_window: bool = False,
        enable_hotkey: bool = True,
        hotkey: str = DEFAULT_HOTKEY,
        command_queue: Optional[Queue] = None,
        ui_focused_queue: Optional[Queue] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            settings: Application settings
            active_window: If True, capture only the active window
            enable_hotkey: If True, enable hotkey toggle for pause/resume
            hotkey: Hotkey string for toggle (default: Ctrl+Shift+P)
            command_queue: Optional queue for receiving commands from UI
            ui_focused_queue: Optional queue for checking if UI is focused
        """
        self.settings = settings
        self.active_window = active_window
        self.enable_hotkey = enable_hotkey
        self.hotkey = hotkey
        self.state = CommentaryState()

        # Command bus queues for UI control
        self._command_queue = command_queue
        self._ui_focused_queue = ui_focused_queue
        self._ui_focused = False

        # Get mode-specific voices for TTS
        alex_voice, morgan_voice = get_voices(settings.commentary_mode)

        # Initialize components
        self.capture = ScreenCapture(
            monitor=settings.capture_monitor,
            active_window=active_window,
        )
        self.vision = VisionAnalyzer(
            api_key=settings.gemini_api_key,
            model=settings.vision_model,
        )
        self.commentary = CommentaryGenerator(
            api_key=settings.gemini_api_key,
            model=settings.vision_model,
            mode=settings.commentary_mode,
        )
        self.tts = TTSManager(
            provider=settings.tts_provider,
            # Gemini settings (default - FREE!) - use mode-specific voices
            gemini_api_key=settings.gemini_api_key,
            gemini_alex_voice=alex_voice,
            gemini_morgan_voice=morgan_voice,
            # ElevenLabs settings (fallback)
            elevenlabs_api_key=settings.elevenlabs_api_key,
            elevenlabs_model=settings.elevenlabs_model,
            elevenlabs_alex_voice=settings.elevenlabs_alex_voice,
            elevenlabs_morgan_voice=settings.elevenlabs_morgan_voice,
            # Dia settings
            dia_model_id=settings.tts_model,
            dia_device=settings.tts_device,
        )
        self.audio = AudioManager(sample_rate=settings.audio_sample_rate)
        self.crowd = CrowdManager(
            sample_rate=settings.audio_sample_rate,
            volume=settings.crowd_volume,
            enabled=settings.crowd_enabled,
            ambient_enabled=settings.crowd_ambient_enabled,
            ambient_volume=settings.crowd_ambient_volume,
        )
        self.sfx = SFXManager(
            sample_rate=settings.audio_sample_rate,
            volume=settings.sfx_volume,
            enabled=settings.sfx_enabled,
        )

        # Hotkey listener
        self._hotkey_listener: Optional[HotkeyListener] = None

        self._running = False
        self._shutdown_requested = False
        self._stop_watch_requested = False
        self._paused_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._state_lock = threading.Lock()

    def _toggle_pause(self) -> None:
        """Toggle pause state (called by hotkey)."""
        is_paused = not self._paused_event.is_set()
        if is_paused:
            self._paused_event.set()
            self.state.paused = True
            self.audio.stop_all()
        else:
            self._paused_event.clear()
            self.state.paused = False
            with self._state_lock:
                self.state.last_screenshot = None
        status = "[yellow]PAUSED[/yellow]" if is_paused else "[green]RESUMED[/green]"
        console.print(f"\n🎙️ Commentary {status} (Press {self.hotkey} to toggle)\n")

    def _poll_commands(self) -> None:
        """Poll for commands from the UI controller."""
        if self._command_queue is None:
            return

        # Process all pending commands
        while True:
            try:
                command = self._command_queue.get_nowait()
            except queue.Empty:
                break

            if command == Command.PAUSE:
                self._paused_event.set()
                self.state.paused = True
                # Stop any currently playing audio immediately
                self.audio.stop_all()
                console.print("\n[yellow]🎙️ Commentary PAUSED (via UI)[/yellow]\n")
            elif command == Command.RESUME:
                self._paused_event.clear()
                self.state.paused = False
                # Clear last screenshot so next capture triggers new commentary
                with self._state_lock:
                    self.state.last_screenshot = None
                console.print("\n[green]🎙️ Commentary RESUMED (via UI)[/green]\n")
            elif command == Command.STOP_COMMENTARY:
                console.print("\n[yellow]🎙️ Stopping commentary (via UI)...[/yellow]\n")
                self._running = False
                self._shutdown_requested = True
                self._shutdown_event.set()
            elif command == Command.STOP_WATCH:
                console.print("\n[red]🎙️ Stop all requested (via UI)...[/red]\n")
                self._running = False
                self._shutdown_requested = True
                self._shutdown_event.set()
                self._stop_watch_requested = True

    def _is_ui_focused(self) -> bool:
        """Check if the UI controller window is currently focused."""
        if self._ui_focused_queue is None:
            return False

        # Get the latest value from the queue
        while True:
            try:
                self._ui_focused = self._ui_focused_queue.get_nowait()
            except queue.Empty:
                break
        return self._ui_focused

    @property
    def stop_watch_requested(self) -> bool:
        """Check if a stop watch command was received."""
        return self._stop_watch_requested

    def setup(self) -> None:
        """Initialize all components."""
        console.print("[bold green]🎙️ ESOPN - AI Sports Commentator Duo[/bold green]")
        console.print("Initializing components...\n")

        with console.status("[bold blue]Loading TTS model..."):
            self.tts.initialize()

        # Align all audio generators to provider output sample rate.
        # This avoids pitch/speed distortions when using Gemini (24kHz).
        provider_rate = self.tts.sample_rate
        self.audio = AudioManager(sample_rate=provider_rate)
        self.crowd = CrowdManager(
            sample_rate=provider_rate,
            volume=self.settings.crowd_volume,
            enabled=self.settings.crowd_enabled,
            ambient_enabled=self.settings.crowd_ambient_enabled,
            ambient_volume=self.settings.crowd_ambient_volume,
        )
        self.sfx = SFXManager(
            sample_rate=provider_rate,
            volume=self.settings.sfx_volume,
            enabled=self.settings.sfx_enabled,
        )

        console.print("[green]✓[/green] TTS model loaded")
        console.print("[green]✓[/green] Vision analyzer ready")
        console.print("[green]✓[/green] Commentary generator ready")
        console.print("[green]✓[/green] Audio player ready")
        console.print(
            "[yellow]![/yellow] Screenshots are sent to the configured Gemini API for analysis"
        )

        # Show commentary mode
        mode_display = {
            "sports": "Sports (ESPN style)",
            "wwe": "WWE Wrestling (JR + King)",
            "freeman_mj": "Freeman + MJ (Calm + Chaos)",
        }
        mode_name = mode_display.get(self.settings.commentary_mode, self.settings.commentary_mode)
        console.print(f"[green]✓[/green] Commentary mode: [bold cyan]{mode_name}[/bold cyan]")

        # Crowd sounds status
        if self.crowd.enabled:
            console.print(
                f"[green]✓[/green] Crowd sounds enabled (volume: {self.settings.crowd_volume:.0%})"
            )
        else:
            console.print("[dim]○[/dim] Crowd sounds disabled")

        # Ambient crowd sounds
        if self.crowd.ambient_enabled:
            console.print(
                f"[green]✓[/green] Ambient crowd enabled (volume: {self.settings.crowd_ambient_volume:.0%})"
            )
        else:
            console.print("[dim]○[/dim] Ambient crowd disabled")

        # Sound effects status
        if self.sfx.enabled:
            console.print(
                f"[green]✓[/green] Sound effects enabled (volume: {self.settings.sfx_volume:.0%})"
            )
        else:
            console.print("[dim]○[/dim] Sound effects disabled")

        # Set up hotkey listener
        if self.enable_hotkey:
            self._hotkey_listener = HotkeyListener(self.hotkey, self._toggle_pause)
            if self._hotkey_listener.start():
                console.print(f"[green]✓[/green] Hotkey ready ({self.hotkey} to pause/resume)")
            else:
                console.print("[yellow]![/yellow] Hotkey not available (pynput issue)")

        if self.active_window:
            console.print("[green]✓[/green] Active window capture mode")

        console.print()

    def run(self) -> None:
        """Run the main commentary loop."""
        self._running = True
        self._setup_signal_handlers()

        mode = "active window" if self.active_window else "full screen"
        console.print(
            f"[bold]Starting commentary ({mode}, every {self.settings.capture_interval}s)[/bold]"
        )
        if self._command_queue is not None:
            console.print(
                "[dim]UI controller active. Use the controller window to pause/stop.[/dim]\n"
            )
        else:
            console.print(f"[dim]Press {self.hotkey} to pause/resume, Ctrl+C to stop[/dim]\n")

        self.audio.start()

        # Start ambient crowd sounds if enabled
        if self.crowd.ambient_enabled:
            ambient_audio = self.crowd.generate_ambient_loop(duration=10.0)
            if ambient_audio is not None:
                self.audio.set_ambient(
                    ambient_audio, volume=1.0
                )  # Volume already applied in CrowdManager
                self.audio.start_ambient()
                console.print("[dim]Ambient crowd sounds started[/dim]")

        try:
            with self.capture:
                while self._running and not self._shutdown_requested:
                    # Poll for UI commands
                    self._poll_commands()

                    if self._paused_event.is_set():
                        time.sleep(0.2)
                        continue

                    if self._is_ui_focused():
                        time.sleep(0.2)
                        continue

                    next_audio, next_scene = self._prepare_commentary()
                    if next_audio is not None and not self._paused_event.is_set():
                        self._speak_commentary_audio(next_audio, next_scene)
                        with self._state_lock:
                            self.state.last_screenshot = None

                    time.sleep(self.settings.capture_interval)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
        finally:
            self.shutdown()

    def _prepare_commentary(self) -> tuple[Optional[SynthesizedAudio], Optional[SceneAnalysis]]:
        """Prepare commentary: capture → analyze → generate → synthesize. Returns audio ready to play."""
        try:
            # 1. Capture screenshot
            screenshot = self._capture_screenshot()
            if screenshot is None:
                return None, None

            # 2. Check if screen has changed enough (skip if <5% difference)
            with self._state_lock:
                last_screenshot = self.state.last_screenshot

            if last_screenshot is not None:
                diff_pct = screenshot.diff_percent(last_screenshot)
                if diff_pct < 5.0:
                    self.state.commentaries_skipped += 1
                    console.print(f"[dim]Screen unchanged ({diff_pct:.1f}% diff), waiting...[/dim]")
                    return None, None
                else:
                    console.print(
                        f"[dim]Screen changed ({diff_pct:.1f}% diff), generating commentary...[/dim]"
                    )

            # Store for next comparison
            with self._state_lock:
                self.state.last_screenshot = screenshot

            # 3. Analyze scene
            scene = self._analyze_scene(screenshot)
            if scene is None:
                return None, None

            # 4. Generate commentary
            commentary = self._generate_commentary(scene)
            if commentary is None:
                return None, None

            # 5. Synthesize audio (but don't play yet)
            audio = self._synthesize_commentary(commentary)
            return audio, scene

        except Exception as e:
            self.state.errors += 1
            logger.error(f"Commentary preparation error: {e}")
            console.print(f"[red]Error: {e}[/red]")
            return None, None

    def _synthesize_commentary(self, commentary: Commentary) -> Optional[SynthesizedAudio]:
        """Synthesize commentary to audio."""
        try:
            audio = self.tts.synthesize(commentary.dialogue)
            return audio
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            self.state.errors += 1
            return None

    def _speak_commentary_audio(
        self, audio: SynthesizedAudio, scene: Optional[SceneAnalysis] = None
    ) -> None:
        """Play pre-synthesized commentary audio with SFX and crowd sounds."""
        try:
            self.state.audio_played += 1

            # Start with the TTS audio
            final_audio = audio.audio
            sfx_info = ""
            crowd_info = ""

            # Check for detected event and get SFX
            if scene and scene.detected_event and self.sfx.enabled:
                sfx = self.sfx.get_sfx_for_event(
                    scene.detected_event,
                    mode=self.settings.commentary_mode,
                )
                if sfx:
                    # Prepend SFX to audio (SFX plays BEFORE commentary)
                    final_audio = self.sfx.create_sfx_then_commentary(
                        sfx,
                        final_audio,
                        commentary_sample_rate=audio.sample_rate,
                        gap_duration=0.2,  # Brief pause for impact
                    )
                    sfx_info = f" [bold yellow]+{sfx.sfx_type.value}[/bold yellow]"

            # Add crowd sounds mixed underneath
            if scene and self.crowd.enabled:
                crowd_audio = self.crowd.get_crowd_audio(
                    intensity=scene.intensity,
                    mood=scene.mood,
                )
                if crowd_audio:
                    final_audio = self.crowd.mix_with_commentary(
                        final_audio,
                        commentary_sample_rate=audio.sample_rate,
                        crowd_audio=crowd_audio,
                        crowd_position="under",
                    )
                    crowd_info = f" [dim]+crowd: {crowd_audio.reaction.value}[/dim]"

            console.print(
                f"[dim]Playing audio ({audio.duration:.1f}s){sfx_info}{crowd_info}...[/dim]"
            )

            # Play synchronously
            self.audio.play_commentary_sync(final_audio, audio.sample_rate)

        except Exception as e:
            logger.error(f"Audio playback failed: {e}")
            self.state.errors += 1
            console.print(f"[red]Audio error: {e}[/red]")

    def _capture_screenshot(self) -> Optional[Screenshot]:
        """Capture a screenshot."""
        try:
            screenshot = self.capture.capture()
            self.state.screenshots_captured += 1

            window_info = f" ({screenshot.window_title})" if screenshot.window_title else ""
            logger.debug(f"Captured: {screenshot.width}x{screenshot.height}{window_info}")
            return screenshot
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            self.state.errors += 1
            return None

    def _analyze_scene(self, screenshot: Screenshot) -> Optional[SceneAnalysis]:
        """Analyze a screenshot."""
        try:
            # Build context from recent scenes
            context = None
            if self.state.recent_scenes:
                context = "; ".join(s.action for s in list(self.state.recent_scenes)[-2:])

            scene = self.vision.analyze(screenshot, context)
            self.state.analyses_completed += 1
            self.state.last_scene = scene

            # Track recent scenes
            self.state.recent_scenes.append(scene)

            # Log the analysis
            event_info = ""
            if scene.detected_event:
                event_info = f" [bold yellow]EVENT: {scene.detected_event}[/bold yellow]"
            console.print(
                f"[dim]Scene:[/dim] {scene.action} "
                f"[dim](mood: {scene.mood}, intensity: {scene.intensity})[/dim]{event_info}"
            )

            return scene

        except Exception as e:
            logger.error(f"Scene analysis failed: {e}")
            self.state.errors += 1
            return None

    def _generate_commentary(self, scene: SceneAnalysis) -> Optional[Commentary]:
        """Generate commentary for a scene."""
        try:
            commentary = self.commentary.generate(scene, list(self.state.recent_scenes))
            self.state.commentaries_generated += 1
            self.state.last_commentary = commentary

            # Display the dialogue
            console.print(
                Panel(
                    commentary.dialogue,
                    title="[bold cyan]Commentary[/bold cyan]",
                    border_style="cyan",
                )
            )

            return commentary

        except Exception as e:
            logger.error(f"Commentary generation failed: {type(e).__name__}: {e}")
            self.state.errors += 1

            # Use fallback commentary
            fallback = get_fallback_commentary(scene.mood)
            console.print(f"[yellow]Using fallback:[/yellow] {fallback}")

            return Commentary(
                dialogue=fallback,
                alex_lines=[],
                morgan_lines=[],
                intensity_used=scene.intensity,
            )

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""

        def handle_shutdown(signum, frame):
            self._shutdown_requested = True
            self._running = False
            self._shutdown_event.set()

        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

    def shutdown(self) -> None:
        """Shutdown all components."""
        console.print("\n[bold]Shutting down...[/bold]")

        self._running = False

        # Stop hotkey listener
        if self._hotkey_listener:
            self._hotkey_listener.stop()

        self.audio.shutdown()
        self.tts.shutdown()

        # Print final stats
        self._print_stats()

    def _print_stats(self) -> None:
        """Print session statistics."""
        table = Table(title="Session Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Uptime", f"{self.state.uptime:.1f}s")
        table.add_row("Screenshots", str(self.state.screenshots_captured))
        table.add_row("Analyses", str(self.state.analyses_completed))
        table.add_row("Commentaries", str(self.state.commentaries_generated))
        table.add_row("Skipped (no change)", str(self.state.commentaries_skipped))
        table.add_row("Audio clips", str(self.state.audio_played))
        table.add_row("Errors", str(self.state.errors))

        console.print(table)


def run_commentary(
    settings: Optional[Settings] = None,
    active_window: bool = False,
    enable_hotkey: bool = True,
    hotkey: str = DEFAULT_HOTKEY,
    command_queue: Optional[Queue] = None,
    ui_focused_queue: Optional[Queue] = None,
) -> None:
    """Run the commentary system with the given settings."""
    if settings is None:
        from .config import get_settings

        settings = get_settings()

    orchestrator = Orchestrator(
        settings,
        active_window=active_window,
        enable_hotkey=enable_hotkey,
        hotkey=hotkey,
        command_queue=command_queue,
        ui_focused_queue=ui_focused_queue,
    )
    orchestrator.setup()
    orchestrator.run()
