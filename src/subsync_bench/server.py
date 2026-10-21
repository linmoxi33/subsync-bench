"""Entry point for the local workbench UI.

    subsync-bench-serve            # http://127.0.0.1:8765
    subsync-bench-serve --port 9000 --open

Binds to loopback only. This tool reads the user's media files and writes
subtitles; exposing it on a LAN interface by default would be a poor choice, so
binding elsewhere requires an explicit --host.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="subsync-bench-serve",
        description="Run the local subtitle workbench UI.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open a browser tab")
    parser.add_argument("--reload", action="store_true", help="dev auto-reload")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        sys.stderr.write(
            "error: the web UI needs extra dependencies.\n"
            "  pip install 'subsync-bench[web]'\n"
        )
        return 3

    from .webapp import create_app

    url = f"http://{args.host}:{args.port}"
    sys.stdout.write(f"subsync-bench workbench -> {url}\n")
    sys.stdout.write("  Ctrl+C to stop\n")
    if args.open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    uvicorn.run(
        create_app() if not args.reload else "subsync_bench.webapp:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=args.reload,
        log_level="warning",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
