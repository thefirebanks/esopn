"""Main orchestrator that coordinates all components for real-time commentary."""

import logging
import signal
import threading
import time
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
    recent_scenes: list[SceneAnalysis] = field(default_factory=list)
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
        )
        self.tts = TTSManager(
            provider=settings.tts_provider,  # type: ignore
            # Gemini settings (default - FREE!)
            gemini_api_key=settings.gemini_api_key,
            gemini_alex_voice=settings.gemini_alex_voice,
            gemini_morgan_voice=settings.gemini_morgan_voice,
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

        # Hotkey listener
        self._hotkey_listener: Optional[HotkeyListener] = None

        self._running = False
        self._shutdown_requested = False
        self._stop_watch_requested = False
        self._pending_cleared = False  # Flag to clear pending commentary on pause

    def _toggle_pause(self) -> None:
        """Toggle pause state (called by hotkey)."""
        self.state.paused = not self.state.paused
        if self.state.paused:
            self.audio.stop_all()
            self._pending_cleared = True
        else:
            self.state.last_screenshot = None  # Force fresh capture on resume
        status = "[yellow]PAUSED[/yellow]" if self.state.paused else "[green]RESUMED[/green]"
        console.print(f"\n🎙️ Commentary {status} (Press {self.hotkey} to toggle)\n")

    def _poll_commands(self) -> None:
        """Poll for commands from the UI controller."""
        if self._command_queue is None:
            return

        # Process all pending commands
        while True:
            try:
                command = self._command_queue.get_nowait()
            except Exception:
                break

            if command == Command.PAUSE:
                self.state.paused = True
                # Stop any currently playing audio immediately
                self.audio.stop_all()
                # Clear any prepared commentary so we start fresh on resume
                self._clear_pending_commentary()
                console.print("\n[yellow]🎙️ Commentary PAUSED (via UI)[/yellow]\n")
            elif command == Command.RESUME:
                self.state.paused = False
                # Clear last screenshot so next capture triggers new commentary
                self.state.last_screenshot = None
                console.print("\n[green]🎙️ Commentary RESUMED (via UI)[/green]\n")
            elif command == Command.STOP_COMMENTARY:
                console.print("\n[yellow]🎙️ Stopping commentary (via UI)...[/yellow]\n")
                self._running = False
                self._shutdown_requested = True
            elif command == Command.STOP_WATCH:
                console.print("\n[red]🎙️ Stop all requested (via UI)...[/red]\n")
                self._running = False
                self._shutdown_requested = True
                self._stop_watch_requested = True
    
    def _clear_pending_commentary(self) -> None:
        """Clear any pending/prepared commentary (used when pausing)."""
        # This will be checked by the main loop
        self._pending_cleared = True

    def _is_ui_focused(self) -> bool:
        """Check if the UI controller window is currently focused."""
        if self._ui_focused_queue is None:
            return False

        # Get the latest value from the queue
        try:
            while not self._ui_focused_queue.empty():
                self._ui_focused = self._ui_focused_queue.get_nowait()
        except Exception:
            pass
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

        console.print("[green]✓[/green] TTS model loaded")
        console.print("[green]✓[/green] Vision analyzer ready")
        console.print("[green]✓[/green] Commentary generator ready")
        console.print("[green]✓[/green] Audio player ready")
        
        # Crowd sounds status
        if self.crowd.enabled:
            console.print(f"[green]✓[/green] Crowd sounds enabled (volume: {self.settings.crowd_volume:.0%})")
        else:
            console.print("[dim]○[/dim] Crowd sounds disabled")
        
        # Ambient crowd sounds
        if self.crowd.ambient_enabled:
            console.print(f"[green]✓[/green] Ambient crowd enabled (volume: {self.settings.crowd_ambient_volume:.0%})")
        else:
            console.print("[dim]○[/dim] Ambient crowd disabled")

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
            console.print("[dim]UI controller active. Use the controller window to pause/stop.[/dim]\n")
        else:
            console.print(f"[dim]Press {self.hotkey} to pause/resume, Ctrl+C to stop[/dim]\n")

        self.audio.start()
        
        # Start ambient crowd sounds if enabled
        if self.crowd.ambient_enabled:
            ambient_audio = self.crowd.generate_ambient_loop(duration=10.0)
            if ambient_audio is not None:
                self.audio.set_ambient(ambient_audio, volume=1.0)  # Volume already applied in CrowdManager
                self.audio.start_ambient()
                console.print("[dim]Ambient crowd sounds started[/dim]")

        try:
            with self.capture:
                import concurrent.futures
                
                # Use a thread pool to prepare next commentary while current plays
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                pending_future = None
                
                while self._running and not self._shutdown_requested:
                    # Poll for UI commands
                    self._poll_commands()
                    
                    # Check if pause cleared pending commentary
                    if self._pending_cleared:
                        if pending_future is not None:
                            pending_future.cancel()
                        pending_future = None
                        self._pending_cleared = False

                    if not self.state.paused:
                        # Check if we have a prepared commentary ready
                        if pending_future is not None and pending_future.done():
                            try:
                                next_audio, next_scene = pending_future.result()
                                pending_future = None
                                
                                if next_audio is not None:
                                    # Play the audio first (blocking)
                                    if not self.state.paused:
                                        self._speak_commentary_audio(next_audio, next_scene)
                                    
                                    # AFTER playback finishes, reset last_screenshot so next
                                    # capture is treated as "changed" - screen likely changed
                                    # during the 5-15 seconds of audio playback
                                    self.state.last_screenshot = None
                                    
                                    # Now start preparing the next commentary
                                    pending_future = executor.submit(self._prepare_commentary)
                                else:
                                    # No change detected and no audio playing - wait before retry
                                    time.sleep(self.settings.capture_interval)
                            except Exception as e:
                                console.print(f"[red]Error getting prepared commentary: {e}[/red]")
                                pending_future = None
                        
                        # If nothing is pending, start preparing immediately
                        if pending_future is None:
                            pending_future = executor.submit(self._prepare_commentary)
                        
                        # Small sleep to avoid busy loop while waiting for prep to complete
                        time.sleep(0.1)
                    else:
                        time.sleep(0.5)  # Check less frequently when paused
                
                executor.shutdown(wait=False)

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
            if self.state.last_screenshot is not None:
                diff_pct = screenshot.diff_percent(self.state.last_screenshot)
                if diff_pct < 5.0:
                    self.state.commentaries_skipped += 1
                    console.print(f"[dim]Screen unchanged ({diff_pct:.1f}% diff), waiting...[/dim]")
                    return None, None
                else:
                    console.print(f"[dim]Screen changed ({diff_pct:.1f}% diff), generating commentary...[/dim]")
            
            # Store for next comparison
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

    def _speak_commentary_audio(self, audio: SynthesizedAudio, scene: Optional[SceneAnalysis] = None) -> None:
        """Play pre-synthesized commentary audio with crowd sounds."""
        try:
            self.state.audio_played += 1

            # Get crowd sounds based on intensity
            final_audio = audio.audio
            crowd_info = ""
            
            if scene and self.crowd.enabled:
                crowd_audio = self.crowd.get_crowd_audio(
                    intensity=scene.intensity,
                    mood=scene.mood,
                )
                if crowd_audio:
                    final_audio = self.crowd.mix_with_commentary(
                        audio.audio, 
                        crowd_audio,
                        crowd_position="under",
                    )
                    crowd_info = f" [dim]+crowd: {crowd_audio.reaction.value}[/dim]"

            console.print(f"[dim]Playing audio ({audio.duration:.1f}s){crowd_info}...[/dim]")

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
                context = "; ".join(s.action for s in self.state.recent_scenes[-2:])

            scene = self.vision.analyze(screenshot, context)
            self.state.analyses_completed += 1
            self.state.last_scene = scene

            # Track recent scenes
            self.state.recent_scenes.append(scene)
            if len(self.state.recent_scenes) > 5:
                self.state.recent_scenes.pop(0)

            # Log the analysis
            console.print(
                f"[dim]Scene:[/dim] {scene.action} "
                f"[dim](mood: {scene.mood}, intensity: {scene.intensity})[/dim]"
            )

            return scene

        except Exception as e:
            logger.error(f"Scene analysis failed: {e}")
            self.state.errors += 1
            return None

    def _generate_commentary(self, scene: SceneAnalysis) -> Optional[Commentary]:
        """Generate commentary for a scene."""
        try:
            commentary = self.commentary.generate(scene, self.state.recent_scenes)
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
