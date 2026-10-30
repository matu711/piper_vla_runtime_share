#!/usr/bin/env python3
"""Wait until the policy endpoint accepts a WebSocket handshake."""

from __future__ import annotations

import argparse
import base64
import os
import socket
import time


def websocket_ready(host: str, port: int, timeout: float) -> None:
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    request = (
        f"GET / HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n"
    ).encode("ascii")
    with socket.create_connection((host, port), timeout=timeout) as connection:
        connection.settimeout(timeout)
        connection.sendall(request)
        status_line = connection.recv(4096).split(b"\r\n", 1)[0]
    if not status_line.startswith(b"HTTP/") or b" 101 " not in status_line:
        raise RuntimeError(f"unexpected WebSocket response: {status_line.decode('ascii', 'replace')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--attempts", type=int, default=120)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()
    if args.attempts < 1 or args.interval < 0 or args.timeout <= 0:
        parser.error("attempts must be positive; interval must be non-negative; timeout must be positive")

    last_error: Exception | None = None
    for attempt in range(1, args.attempts + 1):
        try:
            websocket_ready(args.host, args.port, args.timeout)
        except OSError as exc:
            last_error = exc
        except RuntimeError as exc:
            last_error = exc
        else:
            print(f"Policy WebSocket ready at ws://{args.host}:{args.port} (attempt {attempt})")
            return
        if attempt < args.attempts:
            time.sleep(args.interval)

    raise SystemExit(
        f"Policy WebSocket did not become ready at ws://{args.host}:{args.port} "
        f"after {args.attempts} attempts: {last_error}"
    )


if __name__ == "__main__":
    main()
