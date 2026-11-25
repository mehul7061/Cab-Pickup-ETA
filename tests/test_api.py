import pytest
from fastapi.testclient import TestClient

from app.main import app

BOOKING = {
    "drivers_nearby": 6,
    "distance_to_driver_km": 3.5,
    "hour": 17,
    "rain_mm": 2.0,
    "surge": 1.1,
    "road_closures": 0,
    "car_type": "sedan",
    "zone": "Airport",
}


@pytest.fixture
def client():
    with TestClient(app) as running:
        yield running


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_prediction(client):
    response = client.post("/predict", json=BOOKING)
    assert response.status_code == 200
    assert response.json()["pickup_eta_min"] > 0


@pytest.mark.parametrize("field,value", [
    ("distance_to_driver_km", 15.01), ("rain_mm", -0.01), ("extra", 1),
])
def test_invalid_prediction(client, field, value):
    assert client.post("/predict", json={**BOOKING, field: value}).status_code == 422
