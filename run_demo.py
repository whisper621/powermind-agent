"""Run the PowerMind demo from an IDE's Python Run button."""

from __future__ import annotations

import os
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

from scripts.seed_demo_v1 import ensure_demo_database


ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8000"


def _open_browser_when_ready() -> None:
    for _ in range(90):
        try:
            with urllib.request.urlopen(URL, timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(URL)
                    return
        except OSError:
            time.sleep(1)
    print(f"PowerMind did not become ready. Check the log above, then open {URL}.")


def main() -> None:
    os.chdir(ROOT)
    load_dotenv(ROOT / "config" / ".env")
    ensure_demo_database()
    threading.Thread(target=_open_browser_when_ready, daemon=True).start()

    print(f"PowerMind is starting at {URL}")
    print("Keep this Python Run session active. Stop it with the IDE Stop button.")

    from chainlit.cli import cli

    cli.main(
        args=[
            "run",
            "app_chainlit.py",
            "--headless",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        prog_name="chainlit",
    )


if __name__ == "__main__":
    main()
