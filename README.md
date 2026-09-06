# Motion Tracker

A cross-platform GUI application that captures still images from a camera with configurable exposure (shutter speed), and tracks a user-selected subject across frames using ORB feature matching and homography estimation.

## Features

- **Still image capture** — captures individual frames at intervals determined by the configured shutter speed
- **Configurable exposure** — slider with stepped increments (1ms to 30000ms), applied in real-time via OpenCV's `CAP_PROP_EXPOSURE` (best-effort)
- **Camera selection** — dropdown to select from detected cameras, hot-switchable at any time
- **Click-to-track** — single click on the displayed image to select a subject to track
- **ORB + homography tracking** — robust to rotation and scale changes of the tracked subject
- **Frame-to-frame delta** — displays (dx, dy) displacement between consecutive frames
- **Lost indicator** — clearly shows when tracking is lost; click to re-select a new target
- **Cross-platform** — works on macOS and Windows (Python + Tkinter + OpenCV)

## Requirements

- Python 3.10 or later
- A connected camera (built-in or USB)

## Setup

### macOS

The system Python on macOS ships with an old Tk (8.5) that cannot render images properly. **Use Homebrew Python 3.11+**:

```bash
# Install Python 3.11 and Tk support (if not already installed)
brew install python@3.11 python-tk@3.11

# Clone the repository
git clone <repo-url>
cd motion

# Create a virtual environment with Homebrew Python
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Windows

```bash
# Clone the repository
git clone <repo-url>
cd motion

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python -m motion_tracker
```

1. Select a camera from the dropdown
2. Adjust the exposure slider for your lighting conditions (higher values for dark environments)
3. Click on the subject you want to track in the image
4. The application will display (dx, dy) frame-to-frame displacement
5. If tracking is lost, click again to select a new target

## Running Tests

```bash
python -m pytest tests/ -v
```

Or with unittest:

```bash
python -m unittest discover -s tests -v
```

## Platform Notes

### macOS
- You may need to grant camera permissions when first running the application
- The system may prompt for access in System Settings → Privacy & Security → Camera
- `CAP_PROP_EXPOSURE` behavior varies by camera — the actual exposure may differ from the requested value

### Windows
- Uses DirectShow backend by default
- Exposure values are often interpreted as log2 values (e.g., -4 means 2^-4 = 1/16 second)
- The application sets the value best-effort and displays the actual readback from the camera

## Architecture

```
motion_tracker/
├── __init__.py      # Package init
├── __main__.py      # Entry point
├── camera.py        # Camera capture with exposure control
├── gui.py           # Tkinter GUI
└── tracker.py       # ORB + homography tracker
```

- **camera.py** — runs a capture thread that grabs frames with the configured exposure and pushes them to a queue
- **tracker.py** — ORB feature extraction, BFMatcher with ratio test, RANSAC homography, adaptive reference updates
- **gui.py** — Tkinter window with camera dropdown, exposure slider, image display (click to select target), and status bar
- **__main__.py** — wires everything together and starts the application

## How Tracking Works

1. When you click on the image, the tracker extracts ORB features from an 80×80 pixel region around your click point
2. On each new frame, it detects ORB features in a search region around the last known position
3. Features are matched using a brute-force Hamming distance matcher with Lowe's ratio test
4. A homography is computed via RANSAC to determine the geometric transformation
5. The subject's new center is computed by transforming the reference center through the homography
6. (dx, dy) is the difference between the new center and the previous frame's center
7. Every 10 successful frames, the reference features are refreshed (if confidence is high enough) to adapt to gradual changes in the subject's appearance
