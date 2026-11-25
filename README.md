# MP054 — Cab Pick-up Time (Indore)

SCSE3040 MLOps Mini Project · Group G045 · Batch 22

| Enrolment | Member |
|---|---|
| S24CSEU1309 | Amansingh |
| S24CSEU1324 | MehulChoubey |
| S24CSEU1307 | AbhiRajput |
| S24CSEU1332 | Anshsharma |

Predict booking-to-pickup time in minutes. Evaluate with mean absolute error (MAE).

## Run the Phase 1 dashboard

From the extracted `G045_MP054` folder, run:

```sh
python dashboard.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
Use `python3 dashboard.py` if your system names the Python 3 command `python3`.
Stop the server with Ctrl+C. The dashboard uses only Python's standard library.

Switch between Training, Later data and Holdout to see the date ranges, record
counts, zone distribution and raw CSV rows. Use the zone filter and pagination
to inspect records. The Training audit evidence section shows all seven checks,
computed from the same audit code used by the command-line workflow.

## Stage 1 setup

Open this folder in VS Code and use one public GitHub repository for the group.
If you already have a team repository, copy the project files into its clone,
including the hidden DVC and Git files. Otherwise, run `git init` in this folder.

```sh
python -m venv .venv
```

Activate with `source .venv/bin/activate` on macOS/Linux or
`.venv\Scripts\Activate.ps1` in Windows PowerShell.

```sh
python -m pip install -r requirements.txt
dvc add data/raw/train.csv data/raw/later.csv
dvc status
git status --short
```

The ZIP includes the original CSVs for local use. `dvc add` populates the local
DVC cache. Commit `.dvc/` configuration, `.dvcignore`, `data/raw/.gitignore`,
the two `.dvc` pointers, and `data/raw/holdout_features.csv`. Git ignores
`train.csv` and `later.csv`. To restore the source files in a fresh clone, copy
them from the teacher's ZIP into `data/raw/` or use `dvc pull` if the group has
configured and populated a shared DVC remote. Each member commits their own work.

## Reproduce the data audit

```sh
python -m src.cab_eta.audit
```

The command prints the seven findings and supporting evidence, including zone
summaries, raw and clean category counts, and the target gap and upper tail.
Use `python -m src.cab_eta.audit --write-findings` to refresh the
`data_issues` section of `reports/findings.yaml`.

For the unit mix-up, Old City rain values are recorded at one tenth of the
expected scale. The report stores the recorded-to-correct factor (0.1), so
a later cleaning step would multiply those records by 10. The audit counts
rows in the original training CSV before deduplication.

See PROBLEM.md and STUDENT_HANDBOOK.md for the later training, deployment,
monitoring, and submission requirements.

## Stage 2 — Cleaning, training, and MLflow

Use Python 3.12. After activating your virtual environment, run:

```sh
python -m pip install -r requirements.txt
python -m pytest -q
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) for MLflow. The ZIP includes
the trained v1 model in `models/model.joblib`, the genuine run records in
`mlflow.db`, and the local MLflow artifact directory. Start the combined dashboard with `python dashboard.py` and open
[http://127.0.0.1:8000](http://127.0.0.1:8000). The Model Training link shows
validation MAE and the registered champion directly from `mlflow.db`. The
dashboard runs with Python alone. The standalone MLflow UI is optional.

The training code removes duplicates, corrects the Old City rainfall scale,
treats sentinel values as missing, fixes the target upper tail, and normalizes
car-type spelling. It compares three gradient boosting configurations using a
chronological validation split. The numeric imputer fits only on each run's
training split. The chosen configuration is refit on all cleaned training rows
and registered as `champion`. The later and holdout datasets are reserved for
Stage 4.

To reproduce v1, run `python -m src.cab_eta.train` in a separate extracted copy
with no existing `mlflow.db` or `mlruns/` directory. Running it against the
included database adds more runs and registers a new model version, so use the
included v1 database for Stage 4. The raw CSVs are preserved and the tests use
in-memory fixtures. The FastAPI TestClient test belongs with the Stage 3 API.

Start MLflow from the project root so the bundled relative artifact paths
resolve. If you load the registered model through MLflow's pyfunc interface,
cast the six numeric feature columns to float before calling `predict`.

## Stage 3 — Prediction API and dashboard

From the project folder with Python 3.12, run:

```sh
python -m pip install -r requirements.txt
python -m pytest -q
python -m uvicorn app.main:app
```

Open http://127.0.0.1:8000 for the dashboard and Predict ETA form.
Open http://127.0.0.1:8000/docs for the API explorer. Stop a running
dashboard on port 8000 before starting Uvicorn.

To test the container with Docker Desktop running:

```sh
docker build -t mp054-api:1.0 .
docker run --rm -p 127.0.0.1:8000:8000 mp054-api:1.0
```

Check http://127.0.0.1:8000/health. The API uses the bundled model and
can run without the raw CSVs. For the data audit in a fresh Git clone,
restore the raw CSVs as described in Stage 1. The Kubernetes manifests
are in `k8s/`. GitHub Actions runs the tests on each push.
