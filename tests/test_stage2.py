import pandas as pd
import pytest

from src.cab_eta.clean import FEATURES, clean_training
from src.cab_eta.train import build_model, chronological_split

TARGET = "pickup_eta_min"
FINDINGS = {
    "unit_mixup": {"present": True, "column": "rain_mm", "group_value": "Old City", "factor": 0.1},
    "sentinel_value": {"present": True, "column": "road_closures", "value": 999},
    "target_typos": {"present": True, "count": 1},
}


@pytest.fixture
def raw():
    return pd.DataFrame({
        "date": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03"],
        "zone": ["Old City", "Old City", "Airport", "Tech Park"],
        "car_type": [" min. ", " min. ", "SEDAN", "auto"],
        "rain_mm": [0.2, 0.2, 2.0, 3.0],
        "road_closures": [999, 999, 1, 2],
        "drivers_nearby": [4, 4, 6, 7],
        "distance_to_driver_km": [1.5, 1.5, 1.0, 2.0],
        "hour": [10, 10, 11, 12],
        "surge": [1.0, 1.0, 1.1, 1.2],
        TARGET: [120, 120, 10, 13],
    })


def test_clean_training_uses_audit_without_editing_source(raw):
    original = raw.copy(deep=True)
    cleaned = clean_training(raw, FINDINGS, TARGET)
    assert len(cleaned) == 3
    assert cleaned["car_type"].tolist() == ["mini", "sedan", "auto"]
    assert cleaned["rain_mm"].tolist() == [2.0, 2.0, 3.0]
    assert pd.isna(cleaned.loc[0, "road_closures"])
    assert cleaned[TARGET].tolist() == [12.0, 10.0, 13.0]
    pd.testing.assert_frame_equal(raw, original)


def test_chronological_split_keeps_same_day_together(raw):
    ordered = raw.assign(date=pd.to_datetime([
        "2026-01-01", "2026-01-02", "2026-01-02", "2026-01-03",
    ]))
    train, validation = chronological_split(ordered, 0.5)
    assert len(train) == 1
    assert len(validation) == 3
    assert train["date"].max() < validation["date"].min()


def test_model_predicts_with_unseen_category_and_missing_sensor(raw):
    cleaned = clean_training(raw, FINDINGS, TARGET)
    model = build_model({"n_estimators": 3, "max_depth": 1}, seed=54)
    model.fit(cleaned[FEATURES], cleaned[TARGET])
    example = cleaned[FEATURES].iloc[[0]].assign(car_type="unknown")
    assert float(model.predict(example)[0]) == pytest.approx(12.0, abs=3.0)
