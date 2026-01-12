"""Tkinter-based floating controller window for ESOPN."""

import tkinter as tk
from multiprocessing import Queue
from typing import Optional

from .control import Command


class ControllerWindow:
    """
    Floating Tkinter controller window for ESOPN.

    Provides UI buttons for pause/resume/stop controls as an alternative
    to hotkey-based control. Always stays on top and tracks focus state.
    """

    WINDOW_WIDTH = 240
    WINDOW_HEIGHT = 150
    TITLE = "ESOPN Controller"

    def __init__(self, command_queue: Queue, ui_focused_queue: Queue):
        """
        Initialize the controller window.

        Args:
            command_queue: Queue to send commands to orchestrator
            ui_focused_queue: Queue to report focus state
        """
        self._command_queue = command_queue
        self._ui_focused_queue = ui_focused_queue
        self._root: Optional[tk.Tk] = None
        self._status_var: Optional[tk.StringVar] = None
        self._paused = False

    def _send_command(self, command: Command) -> None:
        """Send a command to the orchestrator."""
        self._command_queue.put(command)

        # Update status based on command
        if command == Command.PAUSE:
            self._paused = True
            self._update_status("Paused")
        elif command == Command.RESUME:
            self._paused = False
            self._update_status("Running")
        elif command == Command.STOP_COMMENTARY:
            self._update_status("Stopping commentary...")
        elif command == Command.STOP_WATCH:
            self._update_status("Stopping all...")

    def _update_status(self, status: str) -> None:
        """Update the status label."""
        if self._status_var:
            self._status_var.set(f"Status: {status}")

    def _on_focus_in(self, event) -> None:
        """Handle window gaining focus."""
        self._report_focus(True)

    def _on_focus_out(self, event) -> None:
        """Handle window losing focus."""
        self._report_focus(False)

    def _report_focus(self, focused: bool) -> None:
        """Report focus state to the orchestrator."""
        # Clear old values and put new state
        try:
            while not self._ui_focused_queue.empty():
                self._ui_focused_queue.get_nowait()
        except Exception:
            pass
        self._ui_focused_queue.put(focused)

    def _on_pause(self) -> None:
        """Handle Pause button click."""
        self._send_command(Command.PAUSE)

    def _on_resume(self) -> None:
        """Handle Resume button click."""
        self._send_command(Command.RESUME)

    def _on_stop_commentary(self) -> None:
        """Handle Stop Commentary button click."""
        self._send_command(Command.STOP_COMMENTARY)
        # Close the window after a short delay
        if self._root:
            self._root.after(500, self._root.destroy)

    def _on_stop_watch(self) -> None:
        """Handle Stop Watch button click."""
        self._send_command(Command.STOP_WATCH)
        # Close the window after a short delay
        if self._root:
            self._root.after(500, self._root.destroy)

    def _on_close(self) -> None:
        """Handle window close button (X)."""
        # Closing the window stops commentary
        self._send_command(Command.STOP_COMMENTARY)
        if self._root:
            self._root.destroy()

    def run(self) -> None:
        """Create and run the controller window."""
        self._root = tk.Tk()
        self._root.title(self.TITLE)

        # Set window size
        self._root.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}")
        self._root.resizable(False, False)

        # Always on top
        self._root.attributes("-topmost", True)

        # Handle window close
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Track focus
        self._root.bind("<FocusIn>", self._on_focus_in)
        self._root.bind("<FocusOut>", self._on_focus_out)

        # Create main frame with padding
        main_frame = tk.Frame(self._root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title label
        title_label = tk.Label(
            main_frame,
            text="ESOPN Controller",
            font=("Helvetica", 12, "bold"),
        )
        title_label.pack(pady=(0, 10))

        # Status label
        self._status_var = tk.StringVar(value="Status: Running")
        status_label = tk.Label(
            main_frame,
            textvariable=self._status_var,
            font=("Helvetica", 10),
            fg="gray",
        )
        status_label.pack(pady=(0, 10))

        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        # Row 1: Pause / Resume
        row1 = tk.Frame(button_frame)
        row1.pack(fill=tk.X, pady=2)

        pause_btn = tk.Button(
            row1,
            text="Pause",
            command=self._on_pause,
            width=10,
        )
        pause_btn.pack(side=tk.LEFT, padx=(0, 5))

        resume_btn = tk.Button(
            row1,
            text="Resume",
            command=self._on_resume,
            width=10,
        )
        resume_btn.pack(side=tk.LEFT)

        # Row 2: Stop button (single, clear action)
        row2 = tk.Frame(button_frame)
        row2.pack(fill=tk.X, pady=2)

        stop_btn = tk.Button(
            row2,
            text="Stop & Exit",
            command=self._on_stop_watch,
            width=22,
            fg="red",
        )
        stop_btn.pack()

        # Center window on screen
        self._root.update_idletasks()
        screen_width = self._root.winfo_screenwidth()
        screen_height = self._root.winfo_screenheight()
        x = (screen_width - self.WINDOW_WIDTH) // 2
        y = (screen_height - self.WINDOW_HEIGHT) // 2
        self._root.geometry(f"+{x}+{y}")

        # Start the main loop
        self._root.mainloop()


def run_controller_window(command_queue: Queue, ui_focused_queue: Queue) -> None:
    """
    Entry point for running the controller window in a separate process.

    Args:
        command_queue: Queue to send commands to orchestrator
        ui_focused_queue: Queue to report focus state
    """
    window = ControllerWindow(command_queue, ui_focused_queue)
    window.run()
