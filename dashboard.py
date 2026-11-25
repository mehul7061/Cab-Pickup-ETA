"""Serve the Cab ETA dashboard locally with Python's standard library."""

import json
import sqlite3
from contextlib import closing
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.cab_eta.audit import audit, read_csv

ROOT = Path(__file__).resolve().parent
DATASETS = ("train", "later", "holdout_features")


def read_training_results():
    database = ROOT / "mlflow.db"
    with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        results = connection.execute("""
            SELECT r.name, m.value AS mae, v.name AS champion_model,
                   v.version AS champion_version
            FROM runs AS r
            JOIN experiments AS e ON e.experiment_id = r.experiment_id
            JOIN latest_metrics AS m ON m.run_uuid = r.run_uuid
            LEFT JOIN model_versions AS v ON v.run_id = r.run_uuid
                AND EXISTS (
                    SELECT 1 FROM registered_model_aliases AS a
                    WHERE a.name = v.name AND a.version = v.version
                      AND a.workspace = v.workspace AND a.alias = 'champion'
                )
            WHERE e.name = ? AND r.lifecycle_stage = 'active'
                AND r.status = 'FINISHED' AND m.key = 'mae_validation' AND m.is_nan = 0
            ORDER BY m.value, r.name
        """, ("MP054 Stage 2",)).fetchall()
    return [dict(run) for run in results]


def read_dashboard_data():
    data_paths = {name: ROOT / "data" / "raw" / f"{name}.csv" for name in DATASETS}
    datasets = {}
    for name, path in data_paths.items():
        columns, rows = read_csv(path)
        datasets[name] = {"columns": columns, "rows": rows}
    findings, _ = audit(
        {**data_paths, "holdout": data_paths["holdout_features"]},
        "pickup_eta_min",
    )
    return {"datasets": datasets, "findings": findings}


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "ui"), **kwargs)

    def do_GET(self):
        if self.path == "/api/training":
            try:
                self.send_json(read_training_results())
            except sqlite3.Error:
                self.send_error(503, "Training results unavailable")
            return
        if self.path != "/api/datasets":
            super().do_GET()
            return

        try:
            data = read_dashboard_data()
        except (OSError, ValueError):
            self.send_error(500, "Unable to load or audit the project CSV files")
            return

        self.send_json(data)

    def send_json(self, data):
        body = json.dumps(data).encode("utf-8")
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
