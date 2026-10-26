"""Local stand-in for claude.ai's `sample` capability, to test Live mode end to end on this machine.

    python scripts/live_proxy.py            # listens on 127.0.0.1:8777

The page's live agent normally asks the viewer's Claude through claude.ai. For a test, a page copy
(web/std_live.html, made by the test harness) gets a small `window.claude` shim that sends each turn
here; this process calls the Messages API with the key in ./.env (never sent to the page) on Haiku,
and returns the reply. Tool calls still run in the page, as they would on claude.ai. Only requests
from the local preview origin are answered.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ORIGIN = "http://127.0.0.1:8765"
MODEL = "claude-haiku-5-5"


def load_env(path: Path = Path(".env")) -> None:
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class Handler(BaseHTTPRequestHandler):
    usage = {"input": 0, "output": 0, "calls": 0}

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", ORIGIN)
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.headers.get("Origin") != ORIGIN or self.path != "/messages":
            self.send_response(403)
            self.end_headers()
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        import anthropic

        client = anthropic.Anthropic()
        req = {k: body[k] for k in ("system", "messages", "tools") if body.get(k)}
        try:
            msg = client.messages.create(model=MODEL, max_tokens=min(int(body.get("max_tokens", 4000)), 8000), **req)
            out = msg.model_dump()
            Handler.usage["input"] += msg.usage.input_tokens
            Handler.usage["output"] += msg.usage.output_tokens
            Handler.usage["calls"] += 1
            code = 200
        except Exception as e:  # report, do not crash the proxy
            out, code = {"error": f"{type(e).__name__}: {str(e)[:300]}"}, 502
        data = json.dumps(out).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        print(json.dumps({"turn": Handler.usage["calls"], "stop": out.get("stop_reason"), "usage": Handler.usage}), flush=True)

    def log_message(self, *a) -> None:
        pass


if __name__ == "__main__":
    load_env()
    ThreadingHTTPServer(("127.0.0.1", 8777), Handler).serve_forever()
