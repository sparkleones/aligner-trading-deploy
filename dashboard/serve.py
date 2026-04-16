"""Dashboard server launcher with SO_REUSEADDR for zombie-proof port binding.

On Windows, zombie processes can hold ports indefinitely after crashing.
This launcher creates a socket with SO_REUSEADDR before passing it to
uvicorn, allowing the dashboard to bind even if ghost processes hold the port.

Note: uvicorn's --fd parameter uses AF_UNIX which doesn't exist on Windows.
Instead, we pre-bind the socket and pass it directly to server.serve(sockets=[...]).

Usage:
    python dashboard/serve.py                  # Default port 8510
    python dashboard/serve.py --port 8512      # Custom port
"""
import asyncio
import os
import sys
import signal
import socket
import argparse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Dashboard server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("DASHBOARD_PORT", 8510)))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--log-level", default="warning")
    args = parser.parse_args()

    # Create socket with SO_REUSEADDR to handle zombie processes holding the port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((args.host, args.port))
    except OSError as e:
        print(f"ERROR: Cannot bind to {args.host}:{args.port} even with SO_REUSEADDR: {e}")
        sys.exit(1)

    sock.listen(128)
    sock.setblocking(False)

    print(f"Dashboard server starting on http://{args.host}:{args.port}/terminal")
    print(f"Socket bound with SO_REUSEADDR on {args.host}:{args.port}")

    # Create uvicorn config WITHOUT fd (fd uses AF_UNIX, not available on Windows)
    config = uvicorn.Config(
        "dashboard.app:app",
        log_level=args.log_level,
    )
    server = uvicorn.Server(config)

    # Ensure socket is properly closed on termination
    def shutdown_handler(signum, frame):
        server.should_exit = True

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, shutdown_handler)

    # Override run() to pass our pre-bound socket directly
    async def serve_with_socket():
        await server.serve(sockets=[sock])

    try:
        asyncio.run(serve_with_socket())
    finally:
        try:
            sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
