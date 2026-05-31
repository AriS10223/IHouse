"""
Launch the i-House Streamlit app.

Usage:
    py run_app.py          (Windows — from any terminal)
    python run_app.py      (Mac / Linux)

What this does:
  1. Creates .venv if it doesn't exist
  2. Installs / updates requirements.txt into the venv
  3. Checks that .env exists
  4. Launches: streamlit run app.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def _run(cmd: list[str], **kwargs) -> int:
    return subprocess.run(cmd, **kwargs).returncode


def _venv_python() -> Path:
    if sys.platform == "win32":
        return ROOT / ".venv" / "Scripts" / "python.exe"
    return ROOT / ".venv" / "bin" / "python"


def _venv_streamlit() -> Path:
    if sys.platform == "win32":
        return ROOT / ".venv" / "Scripts" / "streamlit.exe"
    return ROOT / ".venv" / "bin" / "streamlit"


def ensure_venv():
    venv_python = _venv_python()
    if not venv_python.exists():
        print("[i-House] Creating virtual environment...")
        code = _run([sys.executable, "-m", "venv", str(ROOT / ".venv")])
        if code != 0:
            sys.exit("Failed to create .venv — is Python 3.8+ installed?")
        print("[i-House] Installing dependencies (first run takes ~1 min)...")
        _install_requirements()
    else:
        # Venv exists — sync requirements if requirements.txt is newer than venv
        req_mtime = (ROOT / "requirements.txt").stat().st_mtime
        venv_mtime = venv_python.stat().st_mtime
        if req_mtime > venv_mtime:
            print("[i-House] requirements.txt changed — updating packages...")
            _install_requirements()


def _install_requirements():
    pip = _venv_python()
    code = _run([str(pip), "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")])
    if code != 0:
        sys.exit("pip install failed — check your internet connection and try again.")


def check_env():
    env_path = ROOT / ".env"
    example_path = ROOT / ".env.example"
    if not env_path.exists():
        print()
        print("  ERROR: .env file not found.")
        if example_path.exists():
            print(f"  Copy .env.example to .env and fill in your API keys:")
            print(f"    cp .env.example .env   (Mac/Linux)")
            print(f"    copy .env.example .env (Windows)")
        print()
        sys.exit(1)


def launch():
    streamlit = _venv_streamlit()
    if not streamlit.exists():
        # Fall back to module invocation
        cmd = [str(_venv_python()), "-m", "streamlit", "run", str(ROOT / "app.py"),
               "--server.port", "8501"]
    else:
        cmd = [str(streamlit), "run", str(ROOT / "app.py"), "--server.port", "8501"]

    print()
    print("  i-House is starting...")
    print("  Open http://localhost:8501 in your browser.")
    print("  Press Ctrl+C to stop.")
    print()
    _run(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    ensure_venv()
    check_env()
    launch()
