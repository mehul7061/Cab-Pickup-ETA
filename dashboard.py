"""Serve the Phase 1 dashboard locally with Python's standard library."""

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.cab_eta.audit import audit, read_csv

ROOT = Path(__file__).resolve().parent
DATASETS = ("train", "later", "holdout_features")


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "ui"), **kwargs)

    def do_GET(self):
        if self.path != "/api/datasets":
            super().do_GET()
            return

        try:
            data_paths = {name: ROOT / "data" / "raw" / f"{name}.csv" for name in DATASETS}
            datasets = {}
            for name, path in data_paths.items():
                columns, rows = read_csv(path)
                datasets[name] = {"columns": columns, "rows": rows}
            findings, _ = audit(
                {**data_paths, "holdout": data_paths["holdout_features"]},
                "pickup_eta_min",
            )
        except (OSError, ValueError):
            self.send_error(500, "Unable to load or audit the project CSV files")
            return

        body = json.dumps({"datasets": datasets, "findings": findings}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    with ThreadingHTTPServer(("127.0.0.1", 8000), DashboardHandler) as server:
        print("Cab ETA dashboard: http://127.0.0.1:8000", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
