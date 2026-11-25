"""Measure drift, compare v1 and v2, and export champion predictions."""

import hashlib

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error

from .clean import FEATURES, NUMERIC_FEATURES, clean_training
from .train import ROOT, build_model


def dataset_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def population_stability_index(reference, current, bins):
    """Use equal-width reference bins with open outer edges and smooth empty bins."""
    reference = reference.dropna()
    current = current.dropna()
    low, high = reference.min(), reference.max()
    if low == high:
        low, high = low - 0.5, high + 0.5
    edges = np.linspace(low, high, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf

    def proportions(values):
        counts = np.histogram(values, bins=edges)[0]
        weights = np.maximum(counts / counts.sum(), 1e-6)
        return weights / weights.sum()

    before, after = proportions(reference), proportions(current)
    return float(np.sum((after - before) * np.log(after / before)))


def model_mae(model, frame, target):
    return float(mean_absolute_error(frame[target], model.predict(frame[FEATURES])))


def train_v2(client, name, settings, params, hashes, training, check, target):
    versions = client.search_model_versions(f"name = '{name}'")
    existing = next((version for version in versions if int(version.version) == 2), None)
    if existing:
        prior = client.get_run(existing.run_id)
        if any(prior.data.params.get(key) != value for key, value in hashes.items()):
            raise ValueError("Existing v2 was trained with different source data.")
        return mlflow.sklearn.load_model(f"models:/{name}/2")

    candidate = build_model(params, settings["random_seed"])
    candidate.fit(training[FEATURES], training[target])
    with mlflow.start_run(run_name="retrain-v2"):
        mlflow.log_params({**params, **hashes, "training_rows": len(training),
                           "comparison_rows": len(check)})
        mlflow.log_metric("mae_check", model_mae(candidate, check, target))
        logged = mlflow.sklearn.log_model(
            candidate, name="v2",
            input_example=training[FEATURES].head(3).astype(dict.fromkeys(NUMERIC_FEATURES, float)),
            serialization_format="cloudpickle",
        )
        version = mlflow.register_model(logged.model_uri, name)
    if int(version.version) != 2:
        raise ValueError(f"Expected registered version 2, got {version.version}.")
    return candidate


def main():
    config = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    settings = config["training"]
    monitoring = config["monitoring"]
    paths = {key: ROOT / value for key, value in config["paths"].items()}
    report = yaml.safe_load(paths["findings"].read_text(encoding="utf-8"))
    target = config["target"]
    cleaned = clean_training(pd.read_csv(paths["train"]), report["data_issues"], target)
    later = pd.read_csv(paths["later"], parse_dates=["date"]).sort_values("date", kind="stable")
    split = int(len(later) * monitoring["retrain_fraction"])
    retrain, check = later.iloc[:split], later.iloc[split:]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("MP054 Stage 4")
    client = MlflowClient()
    name = settings["registered_model_name"]
    v1 = mlflow.sklearn.load_model(f"models:/{name}/1")
    psi = {
        column: population_stability_index(cleaned[column], later[column], monitoring["psi_bins"])
        for column in NUMERIC_FEATURES
    }
    mae_by_zone = {zone: model_mae(v1, group, target) for zone, group in later.groupby("zone")}
    v1_mae = model_mae(v1, check, target)
    hashes = {"data_sha256": dataset_hash(paths["train"]), "later_sha256": dataset_hash(paths["later"])}
    model_params = v1[-1].get_params()
    params = {key: model_params[key] for key in settings["candidates"][0]}
    v2 = train_v2(
        client, name, settings, params, hashes,
        pd.concat([cleaned, retrain], ignore_index=True), check, target,
    )
    v2_mae = model_mae(v2, check, target)
    promote = v2_mae <= v1_mae * (1 - monitoring["promotion_improvement"])
    selected_version = 2 if promote else 1
    client.set_registered_model_alias(name, "champion", selected_version)
    champion = mlflow.sklearn.load_model(f"models:/{name}@champion")
    joblib.dump(champion, ROOT / settings["model_path"])

    holdout = pd.read_csv(paths["holdout"])
    predictions = pd.DataFrame({
        "record_id": holdout["record_id"],
        "prediction": champion.predict(holdout[FEATURES]),
    })
    output = ROOT / monitoring["prediction_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)

    report["tracking"]["champion_version"] = selected_version
    report["monitoring"] = {
        "psi": psi,
        "most_drifted_feature": max(psi, key=psi.get),
        "mae_by_group": mae_by_zone,
        "worst_served_group": max(mae_by_zone, key=mae_by_zone.get),
        "mae_v1_on_check": v1_mae,
        "mae_v2_on_check": v2_mae,
        "decision": "promote" if promote else "keep",
    }
    paths["findings"].write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print(f"Comparison rows: {len(check)} | v1 MAE: {v1_mae:.3f} | v2 MAE: {v2_mae:.3f}")
    print(f"Decision: {report['monitoring']['decision']} | Champion: v{selected_version}")
    print(f"Wrote {len(predictions)} predictions to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    import os

    os.chdir(ROOT)
    main()
