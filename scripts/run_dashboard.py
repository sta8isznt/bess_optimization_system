"""Run dashboard."""

import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "dashboards/streamlit_app.py"
    ])
