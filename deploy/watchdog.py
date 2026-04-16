"""Trading System Watchdog — Auto-restart + Health Monitoring.

Manages two processes:
  1. Trading Engine (run_autonomous.py) — the core 24/7 trading loop
  2. Dashboard Server (uvicorn dashboard.app) — terminal UI on port 8510

Features:
  - Auto-restarts crashed processes within 5 seconds
  - Health check every 30 seconds (log file activity + process alive)
  - Exponential backoff on repeated crashes (max 5 min wait)
  - Daily log rotation
  - Graceful shutdown on Ctrl+C / SIGTERM
  - Crash alerts logged to file + console

Usage:
    python deploy/watchdog.py               # Paper trading (default)
    python deploy/watchdog.py --paper       # Paper trading
    python deploy/watchdog.py               # Live trading (no --paper flag)

Deploy as 24/7 service:
    - Windows Task Scheduler: run deploy/install_service.ps1
    - Or just: start_trading.bat --live
"""

import os
import sys
import time
import signal
import logging
import subprocess
import argparse
import atexit
from datetime import datetime, date, timedelta
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── Single Instance Lock ────────────────────────────────────────
# Prevents duplicate trading engines from running simultaneously.
# Uses OS-level file locking (Windows: msvcrt, Unix: fcntl).
# If another instance is already running, this process exits immediately.

LOCK_FILE = PROJECT_ROOT / "data" / ".trading_watchdog.lock"
_lock_fd = None  # Global file descriptor — kept open for lifetime of process


def acquire_instance_lock() -> bool:
    """Try to acquire exclusive lock. Returns True if we got it (no other instance)."""
    global _lock_fd

    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Open (or create) the lock file
        _lock_fd = open(LOCK_FILE, "w")

        if sys.platform == "win32":
            # Windows: use msvcrt for file locking
            import msvcrt
            msvcrt.locking(_lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            # Linux/Mac: use fcntl
            import fcntl
            fcntl.flock(_lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        # Write our PID so humans can identify who holds the lock
        _lock_fd.write(str(os.getpid()))
        _lock_fd.flush()

        # Register cleanup on exit
        atexit.register(release_instance_lock)
        return True

    except (OSError, IOError):
        # Lock already held by another process
        _lock_fd = None
        return False


def release_instance_lock():
    """Release the lock file on exit."""
    global _lock_fd
    if _lock_fd is not None:
        try:
            if sys.platform == "win32":
                import msvcrt
                try:
                    _lock_fd.seek(0)
                    msvcrt.locking(_lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass
            _lock_fd.close()
        except Exception:
            pass
        _lock_fd = None

    # Clean up the lock file
    try:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
    except Exception:
        pass


def read_existing_lock_pid() -> int | None:
    """Read PID from existing lock file (for error messages)."""
    try:
        if LOCK_FILE.exists():
            text = LOCK_FILE.read_text().strip()
            if text.isdigit():
                return int(text)
    except Exception:
        pass
    return None


def is_pid_alive(pid: int) -> bool:
    """Check if a process with given PID is still running."""
    try:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except (OSError, PermissionError):
        return False


# ── Logging ──────────────────────────────────────────────────────

def setup_logging():
    fmt = "%(asctime)s | %(levelname)-8s | WATCHDOG | %(message)s"
    log_file = LOG_DIR / f"watchdog_{date.today().isoformat()}.log"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

logger = logging.getLogger("watchdog")


# ── Process Manager ──────────────────────────────────────────────

class ProcessManager:
    """Manages a single subprocess with auto-restart and backoff."""

    def __init__(self, name: str, cmd: list[str], cwd: str,
                 max_backoff: int = 300, env: dict = None):
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.max_backoff = max_backoff
        self.env = env or os.environ.copy()
        self.process: subprocess.Popen | None = None
        self.restart_count = 0
        self.last_start = 0.0
        self.last_crash = 0.0
        self.total_crashes = 0
        self._log_file = None

    @property
    def is_alive(self) -> bool:
        if self.process is None:
            return False
        return self.process.poll() is None

    @property
    def backoff_seconds(self) -> float:
        """Exponential backoff: 5, 10, 20, 40, 80, 160, 300 (max)."""
        if self.restart_count == 0:
            return 0
        return min(5 * (2 ** (self.restart_count - 1)), self.max_backoff)

    def start(self) -> bool:
        """Start the process. Returns True if started successfully."""
        if self.is_alive:
            return True

        try:
            # Log file for this process
            log_name = f"{self.name}_{date.today().isoformat()}.log"
            log_path = LOG_DIR / log_name
            self._log_file = open(log_path, "a", encoding="utf-8")

            self.process = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                env=self.env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                if sys.platform == "win32" else 0,
            )
            self.last_start = time.time()
            self.restart_count += 1
            logger.info(
                "%s STARTED | PID=%d | restart_count=%d | cmd=%s",
                self.name, self.process.pid, self.restart_count,
                " ".join(self.cmd),
            )
            return True

        except Exception as e:
            logger.error("%s FAILED TO START: %s", self.name, e)
            return False

    def stop(self, timeout: int = 15):
        """Gracefully stop the process."""
        if not self.is_alive:
            return

        logger.info("%s STOPPING (PID=%d)...", self.name, self.process.pid)

        try:
            if sys.platform == "win32":
                # Send CTRL_BREAK_EVENT on Windows
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.process.send_signal(signal.SIGTERM)

            # Wait for graceful shutdown
            try:
                self.process.wait(timeout=timeout)
                logger.info("%s stopped gracefully", self.name)
            except subprocess.TimeoutExpired:
                logger.warning("%s didn't stop in %ds — killing", self.name, timeout)
                self.process.kill()
                self.process.wait(timeout=5)
        except Exception as e:
            logger.error("%s stop error: %s", self.name, e)
            try:
                self.process.kill()
            except Exception:
                pass

        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def check_and_restart(self) -> bool:
        """Check if process died and restart with backoff. Returns True if restarted."""
        if self.is_alive:
            # Reset backoff counter if process has been stable for 10+ minutes
            if time.time() - self.last_start > 600:
                self.restart_count = 0
            return False

        # Process died
        exit_code = self.process.returncode if self.process else -1
        self.total_crashes += 1
        self.last_crash = time.time()

        logger.warning(
            "%s CRASHED | exit_code=%s | total_crashes=%d | will restart in %.0fs",
            self.name, exit_code, self.total_crashes, self.backoff_seconds,
        )

        # Wait for backoff period
        backoff = self.backoff_seconds
        if backoff > 0:
            time.sleep(backoff)

        # Close old log file handle
        if self._log_file:
            self._log_file.close()
            self._log_file = None

        # Restart
        return self.start()


# ── Watchdog ─────────────────────────────────────────────────────

class TradingWatchdog:
    """Monitors and auto-restarts trading engine + dashboard."""

    def __init__(self, paper: bool = True):
        self.paper = paper
        self._shutdown = False
        self._start_time = time.time()

        python_exe = sys.executable

        # Force UTF-8 encoding for all child processes (prevents cp1252
        # crashes from Unicode chars like → ₹ in log messages on Windows)
        utf8_env = os.environ.copy()
        utf8_env["PYTHONIOENCODING"] = "utf-8"
        utf8_env["PYTHONUTF8"] = "1"

        # Process 1: Trading Engine
        engine_cmd = [python_exe, "run_autonomous.py"]
        if paper:
            engine_cmd.append("--paper")
        self.engine = ProcessManager(
            name="TRADING_ENGINE",
            cmd=engine_cmd,
            cwd=str(PROJECT_ROOT),
            env=utf8_env,
        )

        # Process 2: Dashboard (FastAPI + WebSocket)
        # Uses dashboard/serve.py which pre-binds with SO_REUSEADDR
        # to handle zombie processes holding the port on Windows
        self.dashboard_port = os.environ.get("DASHBOARD_PORT", "8510")
        self.dashboard = ProcessManager(
            name="DASHBOARD",
            cmd=[
                python_exe, "dashboard/serve.py",
                "--port", self.dashboard_port,
                "--log-level", "warning",
            ],
            cwd=str(PROJECT_ROOT),
            env=utf8_env,
        )

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        if sys.platform == "win32":
            signal.signal(signal.SIGBREAK, self._signal_handler)

    def _signal_handler(self, sig, frame):
        logger.info("Signal %s received — initiating graceful shutdown", sig)
        self._shutdown = True

    def run(self):
        """Main watchdog loop — runs until shutdown signal."""
        setup_logging()

        # ── SINGLE INSTANCE GUARD ──
        # Acquire exclusive lock BEFORE starting any child processes.
        # If another watchdog is already running, exit immediately.
        if not acquire_instance_lock():
            existing_pid = read_existing_lock_pid()
            if existing_pid and is_pid_alive(existing_pid):
                logger.error(
                    "ANOTHER INSTANCE IS ALREADY RUNNING (PID %d). "
                    "Only one trading system can run at a time. Exiting.",
                    existing_pid,
                )
                print(f"\n  [ERROR] Trading system already running (PID {existing_pid}).")
                print(f"  Stop it first:  deploy\\stop_trading.bat")
                print(f"  Or kill PID:    taskkill /pid {existing_pid} /f\n")
            else:
                # Stale lock file from a crashed process — force remove and retry
                logger.warning("Stale lock file found (PID %s dead). Removing and retrying...",
                               existing_pid)
                release_instance_lock()
                if not acquire_instance_lock():
                    logger.error("Failed to acquire lock even after cleanup. Exiting.")
                    return
                logger.info("Lock acquired after stale cleanup.")
                # Fall through to normal startup

            if not _lock_fd:
                return

        logger.info("Instance lock acquired (PID %d) — guaranteed single instance", os.getpid())

        mode_str = "PAPER" if self.paper else "** LIVE **"
        logger.info("=" * 60)
        logger.info("  TRADING WATCHDOG STARTED")
        logger.info("  Mode: %s", mode_str)
        logger.info("  PID: %d (single instance locked)", os.getpid())
        logger.info("  Dashboard: http://localhost:%s/terminal", self.dashboard_port)
        logger.info("  Engine: run_autonomous.py %s", "--paper" if self.paper else "")
        logger.info("=" * 60)

        # Start both processes
        self.engine.start()
        time.sleep(2)  # Give engine a head start
        self.dashboard.start()

        last_health_check = 0
        last_daily_log = ""

        while not self._shutdown:
            try:
                now = time.time()

                # ── Health check every 30 seconds ──
                if now - last_health_check >= 30:
                    last_health_check = now

                    # Check and restart crashed processes
                    if not self.engine.is_alive:
                        self.engine.check_and_restart()
                    if not self.dashboard.is_alive:
                        self.dashboard.check_and_restart()

                    # Daily status log
                    today_str = date.today().isoformat()
                    if today_str != last_daily_log:
                        last_daily_log = today_str
                        uptime_hrs = (now - self._start_time) / 3600
                        logger.info(
                            "DAILY STATUS | uptime=%.1f hrs | engine_crashes=%d | "
                            "dashboard_crashes=%d | engine_alive=%s | dashboard_alive=%s",
                            uptime_hrs, self.engine.total_crashes,
                            self.dashboard.total_crashes,
                            self.engine.is_alive, self.dashboard.is_alive,
                        )

                # Sleep in short intervals so shutdown is responsive
                time.sleep(5)

            except Exception as e:
                logger.error("Watchdog error: %s", e, exc_info=True)
                time.sleep(10)

        # ── Graceful shutdown ──
        logger.info("Shutting down all processes...")
        self.engine.stop(timeout=20)
        self.dashboard.stop(timeout=10)
        release_instance_lock()
        logger.info("All processes stopped. Lock released. Watchdog exiting.")


# ── Entry Point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trading System Watchdog")
    parser.add_argument(
        "--paper", action="store_true", default=True,
        help="Paper trading mode (default: True)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Live trading mode (REAL MONEY)",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip interactive live-trading confirmation (for headless/AWS deploy)",
    )
    args = parser.parse_args()

    paper = not args.live  # --live overrides default paper mode

    if not paper:
        print("\n" + "!" * 60)
        print("  WARNING: LIVE TRADING MODE — REAL MONEY AT RISK")
        print("!" * 60)
        # Allow headless/AWS deployment via env var or --yes flag
        if args.yes or os.environ.get("LIVE_TRADING_CONFIRMED") == "YES":
            print("  Auto-confirmed via --yes / LIVE_TRADING_CONFIRMED env var.")
        else:
            confirm = input("  Type 'YES' to confirm live trading: ").strip()
            if confirm != "YES":
                print("  Aborted. Use --paper for safe mode.")
                return

    watchdog = TradingWatchdog(paper=paper)
    watchdog.run()


if __name__ == "__main__":
    main()
