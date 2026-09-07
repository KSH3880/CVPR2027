#!/usr/bin/env python3
"""Serve the HTML slide deck and reload the browser when the file changes."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent
DEFAULT_DECK = "tokenhsi_reward_html_slides.html"
RELOAD_ENDPOINT = "/__live_reload__"
RELOAD_SCRIPT = """
<script>
(() => {
  let lastVersion = null;
  setInterval(async () => {
    try {
      const response = await fetch('/__live_reload__?t=' + Date.now(), {
        cache: 'no-store'
      });
      const version = await response.text();
      if (lastVersion !== null && version !== lastVersion) location.reload();
      lastVersion = version;
    } catch (_) {
      // The server may be restarting; retry on the next interval.
    }
  }, 500);
})();
</script>
"""


def make_handler(deck_path: Path):
    class LiveReloadHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(ROOT), **kwargs)

        def do_GET(self):
            request_path = urlsplit(self.path).path

            if request_path == RELOAD_ENDPOINT:
                stat = deck_path.stat()
                payload = f"{stat.st_mtime_ns}:{stat.st_size}".encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            if request_path in {"/", f"/{deck_path.name}"}:
                html = deck_path.read_text(encoding="utf-8")
                html = html.replace("</body>", f"{RELOAD_SCRIPT}</body>")
                payload = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            super().do_GET()

    return LiveReloadHandler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--deck", default=DEFAULT_DECK)
    args = parser.parse_args()

    deck_path = (ROOT / args.deck).resolve()
    if deck_path.parent != ROOT or not deck_path.is_file():
        raise SystemExit(f"Deck not found in {ROOT}: {args.deck}")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(deck_path))
    print(f"Live preview: http://{args.host}:{args.port}/{deck_path.name}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
