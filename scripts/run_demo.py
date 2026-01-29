#!/usr/bin/env python3
"""Script to run the domain adaptation demo."""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the Streamlit demo."""
    demo_path = Path(__file__).parent / "demo" / "app.py"
    
    if not demo_path.exists():
        print(f"Demo file not found: {demo_path}")
        sys.exit(1)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(demo_path)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running demo: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
