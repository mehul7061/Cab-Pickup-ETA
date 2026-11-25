"""Train and register MP054 v1 from the audited training data."""

import hashlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from .clean import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, clean_training

ROOT = Path(__file__).resolve().parents[2]


def chronological_split(frame, validation_fraction):
    split_index = int(len(frame) * (1 - validation_fraction))
    cutoff = frame.iloc[split_index]["date"]
    return frame[frame["date"] < cutoff], frame[frame["date"] >= cutoff]


def build_model(settings, seed):
    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])
    return make_pipeline(
        preprocessing,
        GradientBoostingRegressor(random_state=seed, **settings),
    )


def main():
    config = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    settings = config["training"]
    paths = {key: ROOT / value for key, value in config["paths"].items()}
    report = yaml.safe_load(paths["findings"].read_text(encoding="utf-8"))
    target = config["target"]
    data_hash = hashlib.sha256(paths["train"].read_bytes()).hexdigest()
    cleaned = clean_training(pd.read_csv(paths["train"]), report["data_issues"], target)
    train, validation = chronological_split(cleaned, settings["validation_fraction"])

    # Run from the project root so both SQLite and artifact paths are portable.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    experiment = settings["experiment_name"]
    if client.get_experiment_by_name(experiment) is None:
        client.create_experiment(experiment, artifact_location="mlruns/mp054-stage2")
    mlflow.set_experiment(experiment)

    results = []
    for number, candidate in enumerate(settings["candidates"], start=1):
        model = build_model(candidate, settings["random_seed"])
        with mlflow.start_run(run_name=f"gradient-boosting-{number}") as run:
            mlflow.log_params({
                **candidate, "data_sha256": data_hash,
                "random_seed": settings["random_seed"],
                "validation_start": validation["date"].min().date().isoformat(),
                "train_rows": len(train), "validation_rows": len(validation),
            })
            model.fit(train[FEATURES], train[target])
            mae = mean_absolute_error(validation[target], model.predict(validation[FEATURES]))
            mlflow.log_metric("mae_validation", mae)
            results.append((mae, run.info.run_id, candidate))
            print(f"Candidate {number}: MAE {mae:.3f} min")

    best_mae, run_id, candidate = min(results, key=lambda result: result[0])
    # Refit the selected configuration on all cleaned training records for v1.
    champion = build_model(candidate, settings["random_seed"])
    champion.fit(cleaned[FEATURES], cleaned[target])
    target_path = ROOT / settings["model_path"]
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, target_path)
    with mlflow.start_run(run_id=run_id):
        logged = mlflow.sklearn.log_model(
            champion, name="v1",
            input_example=cleaned[FEATURES].head(3).astype(dict.fromkeys(NUMERIC_FEATURES, float)),
            serialization_format="cloudpickle",
        )
        version = mlflow.register_model(logged.model_uri, settings["registered_model_name"])
    client.set_registered_model_alias(settings["registered_model_name"], "champion", version.version)

    report["tracking"] = {
        "experiment_name": experiment,
        "best_run_id": run_id,
        "registered_model_name": settings["registered_model_name"],
        "champion_version": int(version.version),
    }
    paths["findings"].write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print(f"Champion v{version.version}: validation MAE {best_mae:.3f} min")


if __name__ == "__main__":
    import os

    os.chdir(ROOT)
    main()
