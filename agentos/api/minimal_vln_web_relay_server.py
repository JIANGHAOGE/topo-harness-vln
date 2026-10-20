from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = Path(__file__).resolve().parent / "minimal_vln_web_relay"
ALLOWED_MEDIA_ROOT = (PROJECT_ROOT / "logs" / "minimal_vln").resolve()


def make_handler(tool_url: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/state":
                    self._proxy_state()
                elif parsed.path == "/api/media":
                    self._serve_media(parse_qs(parsed.query).get("path", [""])[0])
                elif parsed.path in {"/", "/index.html"}:
                    self._serve_file(STATIC_ROOT / "index.html")
                elif parsed.path.startswith("/static/"):
                    path = (STATIC_ROOT / parsed.path.removeprefix("/static/")).resolve()
                    if not path.is_relative_to(STATIC_ROOT.resolve()):
                        raise ValueError("invalid static asset path")
                    self._serve_file(path)
                else:
                    self.send_error(404)
            except (OSError, ValueError) as exc:
                self._write_json(400, {"ok": False, "error": str(exc)})

        def log_message(self, format: str, *args: object) -> None:
            if not self.path.startswith("/api/state"):
                print(f"[minimal-vln-relay] {self.address_string()} {format % args}")

        def _proxy_state(self) -> None:
            try:
                with urlopen(f"{tool_url.rstrip('/')}/relay/live", timeout=5.0) as response:
                    body = response.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except URLError as exc:
                self._write_json(503, {"ok": False, "error": f"minimal VLN service unavailable: {exc}"})

        def _serve_media(self, raw_path: str) -> None:
            path = Path(raw_path).expanduser().resolve()
            if not path.is_relative_to(ALLOWED_MEDIA_ROOT):
                raise ValueError("media path is outside logs/minimal_vln")
            self._serve_file(path)

        def _serve_file(self, path: Path) -> None:
            resolved = path.resolve()
            if not resolved.is_file():
                self.send_error(404)
                return
            body = resolved.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(resolved.name)[0] or "application/octet-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _write_json(self, status: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only relay for the isolated minimal VLN experiment.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5915)
    parser.add_argument("--tool-url", default="http://127.0.0.1:8770")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.tool_url))
    print(f"Minimal VLN relay: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
