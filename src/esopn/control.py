"""Command bus for inter-process communication between UI and orchestrator."""

from enum import Enum, auto
from multiprocessing import Queue
from typing import Optional


class Command(Enum):
    """Commands that can be sent from UI to orchestrator."""

    PAUSE = auto()
    RESUME = auto()
    STOP_COMMENTARY = auto()
    STOP_WATCH = auto()


class CommandBus:
    """
    Inter-process communication channel using multiprocessing.Queue.

    Used to send commands from the UI controller to the orchestrator process.
    """

    def __init__(self):
        """Initialize the command bus with a multiprocessing queue."""
        self._command_queue: Queue = Queue()
        self._ui_focused_queue: Queue = Queue()
        self._ui_focused: bool = False

    @property
    def command_queue(self) -> Queue:
        """Get the command queue for passing to subprocesses."""
        return self._command_queue

    @property
    def ui_focused_queue(self) -> Queue:
        """Get the UI focused state queue for passing to subprocesses."""
        return self._ui_focused_queue

    def send_command(self, command: Command) -> None:
        """
        Send a command to the orchestrator.

        Args:
            command: The command to send
        """
        self._command_queue.put(command)

    def get_command(self, timeout: Optional[float] = None) -> Optional[Command]:
        """
        Get a command from the queue (non-blocking by default).

        Args:
            timeout: Optional timeout in seconds. None for non-blocking.

        Returns:
            Command if available, None otherwise
        """
        try:
            if timeout is None:
                return self._command_queue.get_nowait()
            else:
                return self._command_queue.get(timeout=timeout)
        except Exception:
            return None

    def set_ui_focused(self, focused: bool) -> None:
        """
        Update UI focused state.

        Args:
            focused: Whether the UI window is currently focused
        """
        # Clear any old values and put the new state
        try:
            while not self._ui_focused_queue.empty():
                self._ui_focused_queue.get_nowait()
        except Exception:
            pass
        self._ui_focused_queue.put(focused)

    def is_ui_focused(self) -> bool:
        """
        Check if the UI window is currently focused.

        Returns:
            True if UI is focused, False otherwise
        """
        # Get the latest value from the queue
        try:
            while not self._ui_focused_queue.empty():
                self._ui_focused = self._ui_focused_queue.get_nowait()
        except Exception:
            pass
        return self._ui_focused

    def drain(self) -> list[Command]:
        """
        Get all pending commands from the queue.

        Returns:
            List of all pending commands
        """
        commands = []
        while True:
            cmd = self.get_command()
            if cmd is None:
                break
            commands.append(cmd)
        return commands
