# ESOPN - AI Sports Commentator Duo

Real-time AI commentary for coding agents, delivered sports-broadcast style by two AI commentators.

## Features

- **Dual Commentators**: Alex (play-by-play) and Morgan (color commentary) with distinct personalities
- **Real-time Analysis**: Uses Gemini Vision to understand what's happening on screen
- **Natural Dialogue**: Powered by Dia 1.6B for ultra-realistic two-speaker TTS
- **Sports Style**: Exciting, dramatic commentary with gasps, laughs, and emotional reactions
- **Active Window Capture**: Focus on just the terminal/window you're working in
- **Hotkey Toggle**: Pause/resume commentary with Ctrl+Shift+P

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/esopn.git
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

### 2. Run with OpenCode (Recommended)

```bash
# Start commentary + OpenCode together
uv run esopn watch opencode

# Or start commentary first, then use any terminal
uv run esopn watch
# (then open another terminal and run opencode)
```

### 3. Alternative: Manual mode

```bash
# Full screen capture
uv run esopn run

# Capture only the active window
uv run esopn run --active-window

# Shorter interval (more frequent commentary)
uv run esopn run -w -i 3
```

## Usage with OpenCode

The easiest way to use ESOPN with OpenCode:

```bash
# Option 1: Watch mode (recommended)
esopn watch opencode

# Option 2: Run in background
esopn watch &
opencode
```

### Controls

| Action | Key |
|--------|-----|
| Pause/Resume | `Ctrl+Shift+P` |
| Stop | `Ctrl+C` |

## Commands

```bash
esopn run           # Start commentary (full options)
esopn watch         # Watch mode - best for OpenCode
esopn test-capture  # Test screenshot capture
esopn test-tts      # Test TTS synthesis
esopn test-vision   # Test vision analysis
esopn info          # Check system/dependencies
```

### Run Options

```bash
esopn run [OPTIONS]

Options:
  -i, --interval FLOAT      Seconds between screenshots (default: 5.0)
  -m, --monitor INT         Monitor index to capture (default: 1)
  -w, --active-window       Capture only the active/focused window
  -k, --gemini-key TEXT     Google Gemini API key
  -d, --device TEXT         TTS device: cuda, mps, cpu, or auto
  --hotkey TEXT             Hotkey to pause/resume (default: <ctrl>+<shift>+p)
  --no-hotkey               Disable hotkey toggle
  -v, --verbose             Enable verbose logging
```

## How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Screenshot │ ──▶ │   Gemini    │ ──▶ │ Commentary  │ ──▶ │  Dia TTS    │
│  (mss)      │     │   Vision    │     │  Generator  │     │  (1.6B)     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                                                            │
       │                     ┌─────────────┐                       │
       └────────────────────▶│   Speakers  │◀──────────────────────┘
                             └─────────────┘
```

1. **Every 5 seconds** (configurable), captures a screenshot
2. **Gemini Flash** analyzes what's happening (code, terminal, action)
3. **Commentary LLM** generates sports-style dialogue between Alex & Morgan
4. **Dia 1.6B** synthesizes natural two-speaker audio with emotions
5. **Audio plays** through your speakers in real-time

## Requirements

- **Python**: 3.10+
- **API Key**: Google Gemini (free tier works)
- **GPU**: Recommended (~4-5GB VRAM for Dia TTS)
  - NVIDIA CUDA or Apple Silicon MPS
  - CPU works but is slower
- **First run**: Downloads Dia 1.6B model (~4GB)

### macOS Permissions

For active window capture, grant accessibility permissions:
1. System Preferences → Privacy & Security → Accessibility
2. Add your terminal app (Terminal, iTerm2, etc.)

## Commentator Personas

### Alex (S1) - Play-by-Play
- Energetic, fast-paced delivery
- Reacts immediately to action
- Uses sports metaphors
- Quick interjections: "(gasps)", "(laughs)"

**Example**: *"AND THERE IT IS! The agent just nailed that refactor in one shot!"*

### Morgan (S2) - Color Commentary
- Analytical, measured tone
- Provides technical insight
- Dramatic pauses for effect
- Builds on Alex's observations

**Example**: *"You know Alex... I've seen a lot of code in my time, but this recursive solution? Pure elegance."*

## Configuration

Environment variables (prefix with `ESOPN_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | - | Google Gemini API key (required) |
| `ESOPN_CAPTURE_INTERVAL` | 5.0 | Seconds between screenshots |
| `ESOPN_CAPTURE_MONITOR` | 1 | Monitor to capture (1-based) |
| `ESOPN_TTS_DEVICE` | auto | Device: cuda, mps, cpu |
| `ESOPN_VISION_MODEL` | gemini-2.0-flash | Vision model |

Or create a `.env` file:
```bash
GEMINI_API_KEY=your_key_here
ESOPN_CAPTURE_INTERVAL=3.0
```

## Testing Components

```bash
# Test screenshot (saves to file)
uv run esopn test-capture -o screenshot.png
uv run esopn test-capture --active-window -o window.png

# Test TTS (downloads model on first run)
uv run esopn test-tts --play
uv run esopn test-tts -t "[S1] Goal! [S2] (laughs) Incredible!" --play

# Test vision on an image
uv run esopn test-vision screenshot.png
```

## License

MIT
