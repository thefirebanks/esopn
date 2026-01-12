"""Screenshot capture module using mss for cross-platform support."""

import base64
import io
import platform
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import mss
import mss.tools
from PIL import Image


@dataclass
class Screenshot:
    """Represents a captured screenshot."""

    image: Image.Image
    width: int
    height: int
    monitor: int
    timestamp: float
    window_title: Optional[str] = None

    def to_base64(self, format: str = "PNG", max_size: tuple[int, int] = (1024, 768)) -> str:
        """Convert screenshot to base64 string, optionally resizing for API efficiency."""
        img = self.image

        # Resize if larger than max_size to save API tokens
        if img.width > max_size[0] or img.height > max_size[1]:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def to_bytes(self, format: str = "PNG") -> bytes:
        """Convert screenshot to bytes."""
        buffer = io.BytesIO()
        self.image.save(buffer, format=format)
        return buffer.getvalue()
    
    def diff_percent(self, other: "Screenshot") -> float:
        """
        Calculate the percentage of pixels that differ between two screenshots.
        
        Args:
            other: Another screenshot to compare against
            
        Returns:
            Percentage of pixels that differ (0.0 to 100.0)
        """
        import numpy as np
        
        # Resize both to same small size for fast comparison
        size = (256, 256)
        img1 = self.image.convert("L").resize(size, Image.Resampling.BILINEAR)
        img2 = other.image.convert("L").resize(size, Image.Resampling.BILINEAR)
        
        # Convert to numpy arrays
        arr1 = np.array(img1, dtype=np.int16)
        arr2 = np.array(img2, dtype=np.int16)
        
        # Calculate difference - pixels differ if change > threshold
        threshold = 20  # Allow small variations (compression artifacts, etc.)
        diff = np.abs(arr1 - arr2) > threshold
        
        # Return percentage of differing pixels
        return 100.0 * np.sum(diff) / diff.size


def get_active_window_bounds_macos() -> Optional[dict]:
    """
    Get the bounds of the currently focused window on macOS.
    
    Returns:
        Dict with keys: top, left, width, height, title
        None if unable to get window bounds
    
    Note: Requires accessibility permissions for System Events.
    If not granted, will return None and fall back to full screen capture.
    """
    try:
        # First, try to get the frontmost app name (doesn't need accessibility)
        app_script = '''
        tell application "System Events"
            set frontApp to first application process whose frontmost is true
            return name of frontApp
        end tell
        '''
        
        app_result = subprocess.run(
            ["osascript", "-e", app_script],
            capture_output=True,
            text=True,
            timeout=2,
        )
        
        app_name = app_result.stdout.strip() if app_result.returncode == 0 else "Unknown"
        
        # Try to get window bounds (requires accessibility permissions)
        bounds_script = '''
        tell application "System Events"
            set frontApp to first application process whose frontmost is true
            tell frontApp
                set frontWindow to first window
                set windowPosition to position of frontWindow
                set windowSize to size of frontWindow
            end tell
        end tell
        return {item 1 of windowPosition, item 2 of windowPosition, item 1 of windowSize, item 2 of windowSize}
        '''
        
        result = subprocess.run(
            ["osascript", "-e", bounds_script],
            capture_output=True,
            text=True,
            timeout=2,
        )
        
        if result.returncode != 0:
            # Accessibility not granted - return None to fall back to full screen
            return None
        
        # Parse output: "x, y, width, height"
        parts = result.stdout.strip().split(", ")
        if len(parts) != 4:
            return None
        
        x = int(parts[0])
        y = int(parts[1])
        width = int(parts[2])
        height = int(parts[3])
        
        return {
            "left": x,
            "top": y,
            "width": width,
            "height": height,
            "title": app_name,
        }
        
    except (subprocess.TimeoutExpired, ValueError, IndexError, subprocess.SubprocessError):
        return None


def get_active_window_bounds_linux() -> Optional[dict]:
    """
    Get the bounds of the currently focused window on Linux.
    
    Returns:
        Dict with keys: top, left, width, height, title
        None if unable to get window bounds
    """
    try:
        # Get active window ID using xdotool
        result = subprocess.run(
            ["xdotool", "getactivewindow"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        
        if result.returncode != 0:
            return None
        
        window_id = result.stdout.strip()
        
        # Get window geometry
        result = subprocess.run(
            ["xdotool", "getwindowgeometry", "--shell", window_id],
            capture_output=True,
            text=True,
            timeout=2,
        )
        
        if result.returncode != 0:
            return None
        
        # Parse output
        geometry = {}
        for line in result.stdout.strip().split("\n"):
            if "=" in line:
                key, value = line.split("=", 1)
                geometry[key] = value
        
        # Get window name
        result = subprocess.run(
            ["xdotool", "getwindowname", window_id],
            capture_output=True,
            text=True,
            timeout=2,
        )
        
        title = result.stdout.strip() if result.returncode == 0 else "Unknown"
        
        return {
            "left": int(geometry.get("X", 0)),
            "top": int(geometry.get("Y", 0)),
            "width": int(geometry.get("WIDTH", 800)),
            "height": int(geometry.get("HEIGHT", 600)),
            "title": title,
        }
        
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        return None


def get_active_window_bounds() -> Optional[dict]:
    """
    Get the bounds of the currently focused window.
    Cross-platform: supports macOS and Linux.
    
    Returns:
        Dict with keys: top, left, width, height, title
        None if unable to get window bounds
    """
    system = platform.system()
    
    if system == "Darwin":
        return get_active_window_bounds_macos()
    elif system == "Linux":
        return get_active_window_bounds_linux()
    else:
        # Windows or unsupported
        return None


class ScreenCapture:
    """Handles screenshot capture using mss."""

    def __init__(self, monitor: int = 1, active_window: bool = False):
        """
        Initialize screen capture.

        Args:
            monitor: Monitor index (1-based). 0 captures all monitors.
            active_window: If True, capture only the active/focused window.
        """
        self.monitor = monitor
        self.active_window = active_window
        self._sct: Optional[mss.mss] = None

    def __enter__(self) -> "ScreenCapture":
        """Context manager entry."""
        self._sct = mss.mss()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if self._sct:
            self._sct.close()
            self._sct = None

    def capture(self, region: Optional[dict] = None) -> Screenshot:
        """
        Capture a screenshot.

        Args:
            region: Optional dict with keys: top, left, width, height.
                    If None and active_window=True, captures the focused window.
                    If None and active_window=False, captures the entire monitor.

        Returns:
            Screenshot object containing the captured image.
        """
        if self._sct is None:
            self._sct = mss.mss()

        window_title = None

        if region:
            monitor_info = region
        elif self.active_window:
            # Try to get active window bounds
            window_bounds = get_active_window_bounds()
            if window_bounds:
                window_title = window_bounds.pop("title", None)
                monitor_info = window_bounds
            else:
                # Fallback to full monitor
                monitors = self._sct.monitors
                monitor_info = monitors[min(self.monitor, len(monitors) - 1)]
        else:
            # Get monitor geometry
            monitors = self._sct.monitors
            if self.monitor >= len(monitors):
                raise ValueError(
                    f"Monitor {self.monitor} not found. Available: {len(monitors) - 1}"
                )
            monitor_info = monitors[self.monitor]

        # Capture the screen
        sct_img = self._sct.grab(monitor_info)

        # Convert to PIL Image
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

        return Screenshot(
            image=img,
            width=sct_img.width,
            height=sct_img.height,
            monitor=self.monitor,
            timestamp=time.time(),
            window_title=window_title,
        )

    def list_monitors(self) -> list[dict]:
        """List available monitors."""
        if self._sct is None:
            self._sct = mss.mss()
        return self._sct.monitors


def capture_screenshot(monitor: int = 1, active_window: bool = False) -> Screenshot:
    """Convenience function to capture a single screenshot."""
    with ScreenCapture(monitor=monitor, active_window=active_window) as cap:
        return cap.capture()


def capture_active_window() -> Screenshot:
    """Convenience function to capture the active/focused window."""
    return capture_screenshot(active_window=True)
