"""CLI entry point for ESOPN - AI Sports Commentator Duo."""

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from queue import Empty
from typing import Literal, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from . import __version__
from .personas import CommentaryMode

app = typer.Typer(
    name="esopn",
    help="AI Sports Commentator Duo - Real-time commentary for AI coding agents",
    add_completion=False,
)
console = Console()


def _run_commentary_process(
    settings_overrides: dict,
    command_queue,
    ui_focused_queue,
    enable_hotkey: bool,
):
    """Run commentary in a separate process. Must be module-level for pickling."""
    import sys
    import traceback

    try:
        # Re-setup logging for the subprocess
        import logging

        from rich.console import Console
        from rich.logging import RichHandler

        console = Console()
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=console, rich_tracebacks=True)],
            force=True,  # Force reconfiguration in subprocess
        )

        from .config import get_cached_settings
        from .orchestrator import run_commentary

        settings = get_cached_settings()
        settings.capture_interval = settings_overrides["capture_interval"]
        settings.capture_monitor = settings_overrides["capture_monitor"]
        settings.commentary_mode = settings_overrides["commentary_mode"]
        settings.tts_device = settings_overrides["tts_device"]
        settings.gemini_api_key = _resolve_gemini_api_key(settings)

        run_commentary(
            settings,
            active_window=True,  # Always use active window in watch mode
            enable_hotkey=enable_hotkey,
            command_queue=command_queue,
            ui_focused_queue=ui_focused_queue,
        )
    except Exception as e:
        print(f"[ESOPN SUBPROCESS ERROR] {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich output."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def _resolve_gemini_api_key(settings) -> str:
    """Resolve Gemini API key from settings and env with compatibility fallback."""
    key = settings.gemini_api_key
    if not key:
        key = os.environ.get("GEMINI_API_KEY", "")
    return key


def _apply_common_overrides(settings, interval: float, device: str, mode: CommentaryMode) -> None:
    """Apply shared run/watch CLI overrides."""
    settings.capture_interval = interval
    settings.commentary_mode = mode
    if device != "auto":
        settings.tts_device = device


def _validate_output_path(output: Path) -> Path:
    """Ensure output path stays within the current working directory."""
    resolved = output.expanduser().resolve()
    cwd = Path.cwd().resolve()
    if os.path.commonpath([str(cwd), str(resolved)]) != str(cwd):
        raise ValueError(f"Output path must be inside the current directory: {cwd}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


@app.command()
def run(
    interval: float = typer.Option(
        5.0,
        "--interval",
        "-i",
        help="Seconds between screenshots",
    ),
    monitor: int = typer.Option(
        1,
        "--monitor",
        "-m",
        help="Monitor index to capture (1-based)",
    ),
    active_window: bool = typer.Option(
        False,
        "--active-window",
        "-w",
        help="Capture only the active/focused window",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        "-d",
        help="Device for TTS inference: cuda, mps, cpu, or auto",
    ),
    mode: CommentaryMode = typer.Option(
        "sports",
        "--mode",
        "-M",
        help="Commentary mode",
    ),
    hotkey: str = typer.Option(
        "<ctrl>+<shift>+p",
        "--hotkey",
        help="Hotkey to pause/resume commentary",
    ),
    no_hotkey: bool = typer.Option(
        False,
        "--no-hotkey",
        help="Disable hotkey toggle",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """
    Start the AI commentator duo.

    Captures screenshots of your screen and provides real-time
    sports-style commentary on AI coding activity.

    Examples:
        esopn run                           # Full screen capture (sports mode)
        esopn run --active-window           # Capture focused window only
        esopn run -w -i 3                   # Active window, 3s interval
        esopn run --mode wwe                # WWE wrestling style commentary
        esopn run --mode freeman_mj         # Morgan Freeman + Michael Jackson
    """
    setup_logging(verbose)

    # Import here to avoid slow startup for --help
    from .config import get_cached_settings
    from .orchestrator import run_commentary

    # Load settings from .env and environment
    settings = get_cached_settings()

    # Override with CLI options
    _apply_common_overrides(settings, interval=interval, device=device, mode=mode)
    settings.capture_monitor = monitor
    settings.gemini_api_key = _resolve_gemini_api_key(settings)

    # Validate Gemini API key
    if not settings.gemini_api_key:
        console.print(
            "[red]Error: Gemini API key required.[/red]\n"
            "Set ESOPN_GEMINI_API_KEY in your environment or .env file."
        )
        raise typer.Exit(1)

    try:
        run_commentary(
            settings,
            active_window=active_window,
            enable_hotkey=not no_hotkey,
            hotkey=hotkey,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def watch(
    command: Optional[str] = typer.Argument(
        None,
        help="Command to run (e.g., 'opencode', 'vim', 'code'). If not specified, just starts commentary.",
    ),
    interval: float = typer.Option(
        5.0,
        "--interval",
        "-i",
        help="Seconds between screenshots",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        "-d",
        help="Device for TTS inference: cuda, mps, cpu, or auto",
    ),
    mode: CommentaryMode = typer.Option(
        "sports",
        "--mode",
        "-M",
        help="Commentary mode",
    ),
    ui: bool = typer.Option(
        False,
        "--ui",
        help="Show floating controller window instead of hotkey controls",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """
    Watch mode: Start commentary and optionally run a command.

    This is the recommended way to use ESOPN with OpenCode or other tools.
    Commentary captures the active window and runs in the background.

    Examples:
        esopn watch                    # Start commentary, use your terminal
        esopn watch opencode           # Start opencode with commentary
        esopn watch "vim main.py"      # Watch vim session
        esopn watch opencode --ui      # With floating controller window
        esopn watch --ui --mode wwe    # WWE wrestling style!
        esopn watch --mode freeman_mj  # Morgan Freeman + Michael Jackson
    """
    setup_logging(verbose)

    import multiprocessing
    import signal
    import time

    from .config import get_cached_settings

    # Load settings from .env and environment
    settings = get_cached_settings()

    # Override with CLI options
    _apply_common_overrides(settings, interval=interval, device=device, mode=mode)
    settings.capture_monitor = 1
    settings.gemini_api_key = _resolve_gemini_api_key(settings)

    # Validate Gemini API key
    if not settings.gemini_api_key:
        console.print(
            "[red]Error: Gemini API key required.[/red]\n"
            "Set ESOPN_GEMINI_API_KEY in your environment or .env file."
        )
        raise typer.Exit(1)

    # Set up command bus queues if UI is enabled
    command_queue = None
    ui_focused_queue = None
    watch_control_queue = None
    ui_process = None

    if ui:
        from multiprocessing import Queue

        from .ui import run_controller_window

        command_queue = Queue()
        ui_focused_queue = Queue()
        watch_control_queue = Queue()

        # Start UI controller in a separate process
        ui_process = multiprocessing.Process(
            target=run_controller_window,
            args=(command_queue, ui_focused_queue, watch_control_queue),
            daemon=True,
        )

    # Start commentary in background process
    commentary_process = multiprocessing.Process(
        target=_run_commentary_process,
        args=(
            {
                "capture_interval": settings.capture_interval,
                "capture_monitor": settings.capture_monitor,
                "commentary_mode": settings.commentary_mode,
                "tts_device": settings.tts_device,
            },
            command_queue,
            ui_focused_queue,
            not ui,
        ),
        daemon=True,
    )

    console.print("[bold green]🎙️ ESOPN Watch Mode[/bold green]")
    if ui:
        console.print(
            "[dim]UI controller enabled. Use the floating window to control commentary.[/dim]"
        )
    console.print(f"[dim]Starting commentary (active window, every {interval}s)...[/dim]\n")

    # Start processes
    commentary_process.start()
    if ui_process:
        ui_process.start()

    # Give it a moment to initialize
    time.sleep(2)

    # Track the watched command subprocess
    watched_process = None

    def graceful_shutdown_watched():
        """Gracefully shutdown the watched command process."""
        nonlocal watched_process
        if watched_process is not None and watched_process.poll() is None:
            console.print("[yellow]Sending SIGINT to watched process...[/yellow]")
            try:
                watched_process.send_signal(signal.SIGINT)
                # Wait up to 3 seconds for graceful shutdown
                try:
                    watched_process.wait(timeout=3)
                    console.print("[green]Watched process exited gracefully.[/green]")
                except subprocess.TimeoutExpired:
                    console.print(
                        "[yellow]Timeout waiting for graceful shutdown, forcing termination...[/yellow]"
                    )
                    watched_process.terminate()
                    try:
                        watched_process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        watched_process.kill()
                        watched_process.wait()
                    console.print("[red]Watched process terminated.[/red]")
            except Exception as e:
                console.print(f"[red]Error shutting down watched process: {e}[/red]")

    def cleanup_all():
        """Clean up all processes."""
        # Stop watched process if running
        graceful_shutdown_watched()

        # Stop commentary process
        if commentary_process.is_alive():
            commentary_process.terminate()
            commentary_process.join(timeout=2)

        # Stop UI process
        if ui_process and ui_process.is_alive():
            ui_process.terminate()
            ui_process.join(timeout=1)

    def check_stop_watch_command() -> bool:
        """Check if a STOP_WATCH command was received."""
        if watch_control_queue is None:
            return False
        from .control import Command

        while True:
            try:
                cmd = watch_control_queue.get_nowait()
            except Empty:
                break
            if cmd == Command.STOP_WATCH:
                return True
        return False

    if command:
        # Run the specified command
        console.print(f"[bold]Running: {command}[/bold]\n")
        try:
            # Run command in foreground using Popen for more control
            command_args = shlex.split(command)
            if not command_args:
                raise ValueError("Command is empty after parsing")
            watched_process = subprocess.Popen(
                command_args,
                shell=False,
            )

            # Wait for either the command to finish or a stop signal
            while watched_process.poll() is None:
                # Check if commentary process died
                if not commentary_process.is_alive():
                    console.print("\n[yellow]Commentary process exited.[/yellow]")
                    break

                # Check if UI sent STOP_WATCH command
                if ui and check_stop_watch_command():
                    console.print("\n[red]Stop all requested via UI.[/red]")
                    break

                time.sleep(0.5)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
        finally:
            cleanup_all()
    else:
        # No command - just run commentary until Ctrl+C or UI stop
        if ui:
            console.print("[dim]Commentary running. Use the controller window to stop.[/dim]\n")
        else:
            console.print("[dim]Commentary running. Press Ctrl+C to stop.[/dim]\n")
        console.print(
            "[dim]Open another terminal or switch windows to see commentary in action.[/dim]\n"
        )

        try:
            # Wait for either commentary to finish or UI stop command
            while commentary_process.is_alive():
                if ui and check_stop_watch_command():
                    console.print("\n[red]Stop all requested via UI.[/red]")
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping commentary...[/yellow]")
        finally:
            cleanup_all()


@app.command()
def test_capture(
    monitor: int = typer.Option(1, "--monitor", "-m", help="Monitor index"),
    active_window: bool = typer.Option(
        False,
        "--active-window",
        "-w",
        help="Capture only the active window",
    ),
    output: Path = typer.Option(
        Path("test_capture.png"),
        "--output",
        "-o",
        help="Output file path",
    ),
) -> None:
    """Test screenshot capture."""
    from .capture import capture_active_window, capture_screenshot

    if active_window:
        console.print("Capturing active window...")
        screenshot = capture_active_window()
    else:
        console.print(f"Capturing monitor {monitor}...")
        screenshot = capture_screenshot(monitor=monitor)

    try:
        output_path = _validate_output_path(output)
        screenshot.image.save(output_path)
        console.print(f"[green]Saved to {output_path}[/green]")
        console.print(f"Size: {screenshot.width}x{screenshot.height}")
        if screenshot.window_title:
            console.print(f"Window: {screenshot.window_title}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def test_tts(
    text: str = typer.Option(
        "[S1] Welcome to the coding arena! [S2] This is going to be exciting!",
        "--text",
        "-t",
        help="Text to synthesize (use [S1] and [S2] for speakers)",
    ),
    output: Path = typer.Option(
        Path("test_tts.wav"),
        "--output",
        "-o",
        help="Output file path",
    ),
    provider: Literal["auto", "gemini", "elevenlabs", "dia"] = typer.Option(
        "auto",
        "--provider",
        help="TTS provider: gemini, elevenlabs, dia, or auto (uses config)",
    ),
    play: bool = typer.Option(
        False,
        "--play",
        "-p",
        help="Play the audio after synthesis",
    ),
) -> None:
    """Test TTS synthesis."""
    from .config import get_cached_settings
    from .tts import TTSManager

    settings = get_cached_settings()

    # Determine provider
    tts_provider = settings.tts_provider if provider == "auto" else provider
    if tts_provider not in ("gemini", "elevenlabs", "dia"):
        console.print(
            f"[red]Invalid provider: {tts_provider}. Use 'gemini', 'elevenlabs', or 'dia'.[/red]"
        )
        raise typer.Exit(1)

    console.print(f"Using TTS provider: {tts_provider}")

    try:
        tts = TTSManager(
            provider=tts_provider,
            gemini_api_key=_resolve_gemini_api_key(settings),
            gemini_alex_voice=settings.gemini_alex_voice,
            gemini_morgan_voice=settings.gemini_morgan_voice,
            elevenlabs_api_key=settings.elevenlabs_api_key,
            elevenlabs_model=settings.elevenlabs_model,
            elevenlabs_alex_voice=settings.elevenlabs_alex_voice,
            elevenlabs_morgan_voice=settings.elevenlabs_morgan_voice,
            dia_model_id=settings.tts_model,
            dia_device=settings.tts_device,
        )
        tts.initialize()

        console.print(f"Synthesizing: {text}")
        audio = tts.synthesize(text)

        # Save audio
        import soundfile as sf

        output_path = _validate_output_path(output)
        sf.write(output_path, audio.audio, audio.sample_rate)
        console.print(f"[green]Saved to {output_path}[/green]")
        console.print(f"Duration: {audio.duration:.2f}s")

        if play:
            console.print("Playing audio...")
            import sounddevice as sd

            sd.play(audio.audio, audio.sample_rate)
            sd.wait()

        tts.shutdown()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def test_vision(
    image: Path = typer.Argument(..., help="Path to image file"),
) -> None:
    """Test vision analysis on an image."""
    from .config import get_cached_settings

    settings = get_cached_settings()
    gemini_key = _resolve_gemini_api_key(settings)
    if not gemini_key:
        console.print("[red]Error: Gemini API key required.[/red]")
        raise typer.Exit(1)

    if not image.exists():
        console.print(f"[red]Error: File not found: {image}[/red]")
        raise typer.Exit(1)

    from PIL import Image as PILImage

    from .capture import Screenshot
    from .vision import VisionAnalyzer

    console.print(f"Analyzing {image}...")

    try:
        # Load image into Screenshot object
        img = PILImage.open(image)
        screenshot = Screenshot(
            image=img,
            width=img.width,
            height=img.height,
            monitor=0,
            timestamp=0,
        )

        # Analyze
        analyzer = VisionAnalyzer(api_key=gemini_key)
        scene = analyzer.analyze(screenshot)

        console.print(f"\n[bold]Action:[/bold] {scene.action}")
        console.print(f"[bold]Mood:[/bold] {scene.mood}")
        console.print(f"[bold]Intensity:[/bold] {scene.intensity}/10")
        console.print("[bold]Details:[/bold]")
        for detail in scene.details:
            console.print(f"  - {detail}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"esopn {__version__}")


@app.command()
def info() -> None:
    """Show system information and check dependencies."""
    import platform

    console.print("[bold]ESOPN System Information[/bold]\n")

    # System info
    console.print(f"[cyan]Platform:[/cyan] {platform.system()} {platform.machine()}")
    console.print(f"[cyan]Python:[/cyan] {sys.version.split()[0]}")

    # Check for GPU
    console.print("\n[bold]GPU Support:[/bold]")
    try:
        import torch

        console.print(f"[cyan]PyTorch:[/cyan] {torch.__version__}")
        console.print(f"[cyan]CUDA available:[/cyan] {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            console.print(f"[cyan]CUDA device:[/cyan] {torch.cuda.get_device_name(0)}")
        console.print(f"[cyan]MPS available:[/cyan] {torch.backends.mps.is_available()}")
    except ImportError:
        console.print("[yellow]PyTorch not installed[/yellow]")

    # Check key dependencies
    console.print("\n[bold]Dependencies:[/bold]")
    deps = [
        ("mss", "Screenshot capture"),
        ("google.genai", "Vision/LLM API"),
        ("transformers", "TTS model"),
        ("sounddevice", "Audio playback"),
        ("pynput", "Hotkey support"),
        ("PIL", "Image processing"),
        ("rich", "Terminal UI"),
    ]
    for dep, desc in deps:
        try:
            __import__(dep.split(".")[0])
            console.print(f"[green]✓[/green] {dep} ({desc})")
        except ImportError:
            console.print(f"[red]✗[/red] {dep} ({desc})")

    # Check accessibility permissions (macOS)
    if platform.system() == "Darwin":
        console.print("\n[bold]macOS Permissions:[/bold]")
        console.print("[dim]Active window capture requires accessibility permissions.[/dim]")
        console.print(
            "[dim]Grant access in System Preferences → Privacy & Security → Accessibility[/dim]"
        )


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
