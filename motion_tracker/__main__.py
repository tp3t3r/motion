"""Entry point for the Motion Tracker application.

Usage:
    python -m motion_tracker
"""

import logging
import sys


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr
    )
    
    from .gui import TrackerApp
    
    app = TrackerApp()
    app.run()


if __name__ == "__main__":
    main()
