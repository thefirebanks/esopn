# ESOPN - AI Sports Commentator Duo

Real-time AI commentary for coding sessions, delivered sports-broadcast style by two AI commentators.

## Features

- **Dual Commentators**: Alex (play-by-play) and Morgan (color commentary) with distinct personalities
- **Real-time Analysis**: Uses Gemini Vision to understand what's happening on screen
- **Natural Dialogue**: Powered by Gemini TTS for realistic two-speaker audio (free!)
- **Floating UI Controller**: Pause, resume, or stop commentary with a simple floating window
- **Smart Change Detection**: Only comments when the screen actually changes
- **Hotkey Toggle**: Pause/resume commentary with Ctrl+Shift+P

## Installation

```bash
# Clone the repo
git clone https://github.com/thefirebanks/esopn.git
cd esopn

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### 1. Set up your API key

```bash
# Get a free API key from https://aistudio.google.com/apikey
export GEMINI_API_KEY=your_key_here
```

### 2. Run with the UI controller (Recommended)

```bash
# Start commentary with floating UI controller
uv run esopn watch --ui
```

This launches a floating window with Pause, Resume, and Stop buttons. Switch to your code editor or terminal and watch the commentary roll in!

### 3. Alternative: Headless mode

```bash
# Run without UI (use Ctrl+Shift+P to pause, Ctrl+C to stop)
uv run esopn watch
```

## Controls

| Action | Method |
|--------|--------|
| Pause/Resume | Click button in UI, or press `Ctrl+Shift+P` |
| Stop | Click "Stop & Exit" in UI, or press `Ctrl+C` |

## Commands

```bash
esopn watch         # Start commentary (recommended)
esopn watch --ui    # Start with floating UI controller
esopn run           # Start commentary (full options)
esopn test-capture  # Test screenshot capture
esopn test-tts      # Test TTS synthesis
esopn test-vision   # Test vision analysis
esopn info          # Check system/dependencies
```

### Watch Options

```bash
esopn watch [OPTIONS]

Options:
  --ui                      Show floating UI controller window
  -w, --active-window       Capture only the active/focused window (default: on)
  -k, --gemini-key TEXT     Google Gemini API key
  --hotkey TEXT             Hotkey to pause/resume (default: <ctrl>+<shift>+p)
  --no-hotkey               Disable hotkey toggle
  -v, --verbose             Enable verbose logging
```

## How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Screenshot │ ──▶ │   Gemini    │ ──▶ │ Commentary  │ ──▶ │ Gemini TTS  │
│  (mss)      │     │   Vision    │     │  Generator  │     │  (free!)    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                                                            │
       │                     ┌─────────────┐                       │
       └────────────────────▶│   Speakers  │◀──────────────────────┘
                             └─────────────┘
```

1. **Captures screenshots** and detects when the screen changes (>5% difference)
2. **Gemini Vision** analyzes what's happening (code, terminal, action)
3. **Commentary LLM** generates sports-style dialogue between Alex & Morgan
4. **Gemini TTS** synthesizes natural two-speaker audio with distinct voices
5. **Audio plays** through your speakers in real-time

## Requirements

- **Python**: 3.10+
- **API Key**: Google Gemini (free tier works great!)
- **macOS/Linux**: For screen capture

### macOS Permissions

For screen capture, grant permissions:
1. System Preferences → Privacy & Security → Screen Recording
2. Add your terminal app (Terminal, iTerm2, etc.)

For active window capture, also grant:
1. System Preferences → Privacy & Security → Accessibility
2. Add your terminal app

## Commentator Personas

### Alex (S1) - Play-by-Play
- High-energy, describes what's happening
- Calls out specific actions, file names, patterns
- Uses conversational descriptions (not literal code reading)

**Example**: *"New submit handler going in! Looks like they're setting up form validation!"*

### Morgan (S2) - Color Commentary  
- Analytical with energy and enthusiasm
- Explains WHY the code matters
- Provides technical insight

**Example**: *"That's the Strategy pattern right there - makes it easy to swap algorithms later!"*

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | - | Google Gemini API key (required) |
| `ESOPN_CAPTURE_INTERVAL` | 3.0 | Seconds between screenshots |
| `ESOPN_TTS_PROVIDER` | gemini | TTS provider (gemini or elevenlabs) |

Or create a `.env` file:
```bash
GEMINI_API_KEY=your_key_here
ESOPN_CAPTURE_INTERVAL=3.0
```

## License

MIT
