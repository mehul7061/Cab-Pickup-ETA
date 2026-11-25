"""Serve cab ETA predictions and the project dashboard."""

import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from dashboard import ROOT, read_dashboard_data, read_training_results
from src.cab_eta.clean import FEATURES

logger = logging.getLogger("cab_eta.predictions")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.propagate = False


class Booking(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    drivers_nearby: int = Field(ge=0)
    distance_to_driver_km: float = Field(ge=0, le=15)
    hour: int = Field(ge=0, le=23)
    rain_mm: float = Field(ge=0)
    surge: float = Field(gt=0)
    road_closures: int = Field(ge=0)
    car_type: Literal["auto", "mini", "sedan"]
    zone: Literal["Airport", "Old City", "Tech Park"]


@asynccontextmanager
async def lifespan(application):
    application.state.model = joblib.load(ROOT / "models" / "model.joblib")
    yield


app = FastAPI(title="Cab Pickup ETA", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(booking: Booking):
    features = pd.DataFrame([booking.model_dump()], columns=FEATURES)
    eta = round(float(app.state.model.predict(features)[0]), 3)
    logger.info(json.dumps({"event": "prediction", "pickup_eta_min": eta}))
    return {"pickup_eta_min": eta}


@app.get("/api/datasets")
def datasets():
    try:
        return read_dashboard_data()
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail="Local datasets unavailable") from error


@app.get("/api/training")
def training():
    return read_training_results()


@app.get("/api/monitoring")
def monitoring():
    report = yaml.safe_load((ROOT / "reports" / "findings.yaml").read_text(encoding="utf-8"))
    return {
        **report["monitoring"],
        "champion_version": report["tracking"]["champion_version"],
    }


app.mount("/", StaticFiles(directory=ROOT / "ui", html=True), name="dashboard")
