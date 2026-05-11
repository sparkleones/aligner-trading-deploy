"""
Local laptop launcher for the Aligner Trading System (Windows/Mac/Linux).

Runs both processes:
  1. Dashboard (FastAPI on port 8510)
  2. Autonomous trading engine

Logs from both stream to this terminal. Ctrl+C stops both gracefully.

Singleton: refuses to start if another launcher OR engine OR something
holding port 8510 is already running. Pass --force to override (will
kill the other instance first).

Usage:
    python start_local.py            # Live trading (uses .env PAPER_TRADING)
    python start_local.py --paper    # Force paper mode
    python start_local.py --port 8510
    python start_local.py --force    # Kill existing instance and restart

Pre-requisites:
  - .env file with at least: BROKER_API_KEY, BROKER_API_SECRET
  - For auto-login: ZERODHA_USER_ID, ZERODHA_PASSWORD, ZERODHA_TOTP_SECRET
  - python -m pip install -r requirements.txt
"""
import os
import sys
import signal
import socket
import subprocess
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LAUNCHER_LOCK = PROJECT_ROOT / "data" / "launcher.lock"
ENGINE_LOCK = PROJECT_ROOT / "data" / "engine.lock"

# Ensure imports work for child processes
os.environ["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("TZ", "Asia/Kolkata")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8510, help="Dashboard port (default 8510)")
    p.add_argument("--paper", action="store_true", help="Force paper trading mode")
    p.add_argument("--no-engine", action="store_true",
                   help="Only run dashboard (skip engine — useful for first-time auth)")
    p.add_argument("--force", action="store_true",
                   help="Kill any existing launcher/engine before starting")
    return p.parse_args()


def banner(msg):
    bar = "=" * 70
    print(f"\n+{bar}+")
    print(f"| {msg:<68} |")
    print(f"+{bar}+\n", flush=True)


def port_in_use(port: int) -> bool:
    """True if something is already bound to TCP port on localhost."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        result = s.connect_ex(("127.0.0.1", port))
        return result == 0
    except OSError:
        return False
    finally:
        s.close()


def pid_is_alive(pid: int) -> bool:
    """Check whether a PID is currently running (cross-platform)."""
    if pid <= 0:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        pass
    # Fallback: best-effort
    if os.name == "nt":
        try:
            out = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}"], stderr=subprocess.DEVNULL
            ).decode("utf-8", errors="replace")
            return str(pid) in out
        except Exception:
            return True
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def read_lock_pid(path: Path) -> int:
    """Return PID stored in a lock file, or 0 if missing/unreadable."""
    try:
        if not path.exists():
            return 0
        txt = path.read_text(encoding="utf-8").strip()
        # First non-empty integer token
        for tok in txt.split():
            try:
                return int(tok)
            except ValueError:
                continue
    except Exception:
        pass
    return 0


def kill_pid(pid: int) -> None:
    if pid <= 0:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    else:
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


def preflight_singleton(port: int, force: bool) -> int:
    """
    Refuse to launch if a launcher, engine, or port is already in use.
    With --force, kill those first and continue.
    Returns 0 on clean to proceed, non-zero on hard exit.
    """
    issues = []

    # 1. Existing launcher
    launcher_pid = read_lock_pid(LAUNCHER_LOCK)
    if launcher_pid and pid_is_alive(launcher_pid):
        issues.append(("launcher", launcher_pid))

    # 2. Existing engine
    engine_pid = read_lock_pid(ENGINE_LOCK)
    if engine_pid and pid_is_alive(engine_pid):
        issues.append(("engine", engine_pid))

    # 3. Port in use
    if port_in_use(port):
        issues.append(("dashboard port", f"{port} (already bound)"))

    if not issues:
        # Stale locks → clean
        for p in (LAUNCHER_LOCK, ENGINE_LOCK):
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
        return 0

    print()
    print("=" * 70)
    print("  REFUSING TO START — another instance is already running")
    print("=" * 70)
    for kind, pid in issues:
        print(f"    {kind}: PID {pid}")
    print()

    if not force:
        print("  Options:")
        print("    1. Switch to the existing launcher window and use it.")
        print("    2. Kill it cleanly:")
        if os.name == "nt":
            print('       Get-Process python | Stop-Process -Force')
        else:
            print("       pkill -f 'python.*start_local.py'")
            print("       pkill -f 'python.*run_autonomous.py'")
        print("    3. Re-run with --force to auto-kill and restart:")
        print(f"       python start_local.py --force")
        print()
        return 2

    # --force: kill them
    print("  --force given. Killing existing processes...", flush=True)
    for kind, pid in issues:
        if isinstance(pid, int):
            print(f"    killing {kind} PID {pid}", flush=True)
            kill_pid(pid)
    # Also remove lock files
    for p in (LAUNCHER_LOCK, ENGINE_LOCK):
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass
    # Give OS a moment to release the port
    time.sleep(2)
    if port_in_use(port):
        print(f"  Port {port} still in use after kill — exiting.", flush=True)
        return 3
    print("  OK — continuing.", flush=True)
    return 0


def write_launcher_lock():
    try:
        LAUNCHER_LOCK.parent.mkdir(parents=True, exist_ok=True)
        LAUNCHER_LOCK.write_text(str(os.getpid()), encoding="utf-8")
    except Exception:
        pass


def clear_launcher_lock():
    try:
        if LAUNCHER_LOCK.exists():
            LAUNCHER_LOCK.unlink()
    except Exception:
        pass


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

    # Singleton preflight
    rc = preflight_singleton(args.port, args.force)
    if rc != 0:
        return rc

    write_launcher_lock()

    banner("ALIGNER TRADING — LOCAL LAUNCHER")
    print(f"  Project:    {PROJECT_ROOT}")
    print(f"  Launcher:   PID {os.getpid()} (lock: {LAUNCHER_LOCK.name})")
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
    engine_args = None
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
        clear_launcher_lock()
        print("  All processes stopped.", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown)

    # ── Watch loop: restart dead processes ──
    # Bounded restart attempts to avoid crash-loop hell
    dashboard_restarts = 0
    engine_restarts = 0
    MAX_RESTARTS = 5
    try:
        while True:
            time.sleep(10)
            if dashboard_proc.poll() is not None:
                if dashboard_restarts >= MAX_RESTARTS:
                    print(f"\n[FATAL] Dashboard has died {dashboard_restarts} times — "
                          f"stopping launcher to avoid crash-loop.", flush=True)
                    shutdown()
                dashboard_restarts += 1
                print(f"\n[WARN] Dashboard died (exit {dashboard_proc.returncode}). "
                      f"Restart {dashboard_restarts}/{MAX_RESTARTS}...", flush=True)
                # Sanity: don't restart if port is now held by something else
                if port_in_use(args.port):
                    print(f"[FATAL] Port {args.port} held by another process — exiting.", flush=True)
                    shutdown()
                dashboard_proc = subprocess.Popen(
                    dashboard_cmd, cwd=str(PROJECT_ROOT),
                    stdout=sys.stdout, stderr=sys.stderr,
                )
                print(f"  New dashboard PID: {dashboard_proc.pid}", flush=True)
            if engine_proc is not None and engine_proc.poll() is not None:
                if engine_restarts >= MAX_RESTARTS:
                    print(f"\n[FATAL] Engine has died {engine_restarts} times — "
                          f"stopping launcher.", flush=True)
                    shutdown()
                engine_restarts += 1
                print(f"\n[WARN] Engine died (exit {engine_proc.returncode}). "
                      f"Restart {engine_restarts}/{MAX_RESTARTS} in 10s...", flush=True)
                time.sleep(10)
                # If another engine has grabbed the lock in the meantime, exit
                other_engine = read_lock_pid(ENGINE_LOCK)
                if other_engine and pid_is_alive(other_engine) and other_engine != engine_proc.pid:
                    print(f"[FATAL] Another engine (PID {other_engine}) already running — exiting.", flush=True)
                    shutdown()
                engine_proc = subprocess.Popen(
                    engine_args, cwd=str(PROJECT_ROOT),
                    stdout=sys.stdout, stderr=sys.stderr,
                )
                print(f"  New engine PID: {engine_proc.pid}", flush=True)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
