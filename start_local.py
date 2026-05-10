"""
Local laptop launcher for the Aligner Trading System (Windows/Mac/Linux).

Runs both processes:
  1. Dashboard (FastAPI on port 8510)
  2. Autonomous trading engine

Logs from both stream to this terminal. Ctrl+C stops both gracefully.

Equivalent to deploy/start.sh used inside the Docker container, but
adapted for direct execution on a laptop.

Usage:
    python start_local.py            # Live trading (uses .env PAPER_TRADING)
    python start_local.py --paper    # Force paper mode
    python start_local.py --port 8510

Pre-requisites:
  - .env file with at least: BROKER_API_KEY, BROKER_API_SECRET
  - For auto-login: ZERODHA_USER_ID, ZERODHA_PASSWORD, ZERODHA_TOTP_SECRET
  - python -m pip install -r requirements.txt
"""
import os
import sys
import signal
import subprocess
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Ensure imports work for child processes
os.environ["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("TZ", "Asia/Kolkata")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8510, help="Dashboard port (default 8510)")
    p.add_argument("--paper", action="store_true", help="Force paper trading mode")
    p.add_argument("--no-engine", action="store_true",
                   help="Only run dashboard (skip engine — useful for first-time auth)")
    return p.parse_args()


def banner(msg):
    bar = "=" * 70
    print(f"\n+{bar}+")
    print(f"| {msg:<68} |")
    print(f"+{bar}+\n", flush=True)


def main():
    args = parse_args()

    # Soft-validate .env
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        print(f"[ERROR].env not found at {env_path}")
        return 1
    env_text = env_path.read_text(encoding="utf-8", errors="replace")
    required = ["BROKER_API_KEY", "BROKER_API_SECRET"]
    missing = [k for k in required if f"\n{k}=" not in "\n" + env_text]
    if missing:
        print(f"[ERROR]Missing required env vars: {', '.join(missing)}")
        return 1

    banner("ALIGNER TRADING — LOCAL LAUNCHER")
    print(f"  Project:    {PROJECT_ROOT}")
    print(f"  Dashboard:  http://localhost:{args.port}/terminal")
    print(f"  API check:  http://localhost:{args.port}/api/broker/auto_login/check")
    print(f"  Paper mode: {args.paper or 'from .env'}")
    print(f"  Engine:     {'DISABLED (--no-engine)' if args.no_engine else 'enabled'}")
    print()

    # ── Spawn dashboard ──
    print("Starting dashboard...", flush=True)
    dashboard_cmd = [
        sys.executable, "-m", "uvicorn",
        "dashboard.app:app",
        "--host", "0.0.0.0",
        "--port", str(args.port),
        "--log-level", "warning",
    ]
    dashboard_proc = subprocess.Popen(
        dashboard_cmd, cwd=str(PROJECT_ROOT),
        stdout=sys.stdout, stderr=sys.stderr,
    )
    print(f"  Dashboard PID: {dashboard_proc.pid}", flush=True)

    # Wait for dashboard to come up
    print("  Waiting 5s for dashboard...", flush=True)
    time.sleep(5)

    # ── Spawn engine (unless --no-engine) ──
    engine_proc = None
    if not args.no_engine:
        print("\nStarting autonomous trading engine...", flush=True)
        engine_args = [sys.executable, "run_autonomous.py"]
        if args.paper:
            engine_args.append("--paper")
        engine_proc = subprocess.Popen(
            engine_args, cwd=str(PROJECT_ROOT),
            stdout=sys.stdout, stderr=sys.stderr,
        )
        print(f"  Engine PID: {engine_proc.pid}", flush=True)
    else:
        print("\nEngine NOT started (--no-engine).", flush=True)
        print(f"To trigger auth, open: http://localhost:{args.port}/terminal")
        print("Then click AUTO-LOGIN in the Kite Broker panel.\n", flush=True)

    banner(f"BOTH RUNNING — open http://localhost:{args.port}/terminal")
    print("Press Ctrl+C to stop both.\n", flush=True)

    # ── Signal handler for clean shutdown ──
    def shutdown(signum=None, frame=None):
        print("\nShutting down...", flush=True)
        if engine_proc and engine_proc.poll() is None:
            print("  Stopping engine...", flush=True)
            try:
                engine_proc.terminate()
                engine_proc.wait(timeout=10)
            except Exception:
                engine_proc.kill()
        if dashboard_proc.poll() is None:
            print("  Stopping dashboard...", flush=True)
            try:
                dashboard_proc.terminate()
                dashboard_proc.wait(timeout=5)
            except Exception:
                dashboard_proc.kill()
        print("  All processes stopped.", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown)

    # ── Watch loop: restart dead processes ──
    try:
        while True:
            time.sleep(10)
            if dashboard_proc.poll() is not None:
                print(f"\n[WARN]Dashboard died (exit {dashboard_proc.returncode}). Restarting...", flush=True)
                dashboard_proc = subprocess.Popen(
                    dashboard_cmd, cwd=str(PROJECT_ROOT),
                    stdout=sys.stdout, stderr=sys.stderr,
                )
                print(f"  New dashboard PID: {dashboard_proc.pid}", flush=True)
            if engine_proc is not None and engine_proc.poll() is not None:
                print(f"\n[WARN]Engine died (exit {engine_proc.returncode}). Restarting in 10s...", flush=True)
                time.sleep(10)
                engine_proc = subprocess.Popen(
                    engine_args, cwd=str(PROJECT_ROOT),
                    stdout=sys.stdout, stderr=sys.stderr,
                )
                print(f"  New engine PID: {engine_proc.pid}", flush=True)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
