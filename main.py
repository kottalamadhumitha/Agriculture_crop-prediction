import json
import os
import pickle
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from apy_loader import get_apy_repository
from features import feature_matrix_from_inputs
from geocode import reverse_geocode_osm

ROOT = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    """Optional .env next to main.py (no extra dependency)."""
    p = ROOT / ".env"
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()

app = FastAPI(title="Crop Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open(ROOT / "model.pkl", "rb") as f:
    model = pickle.load(f)
with open(ROOT / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(ROOT / "imputer.pkl", "rb") as f:
    imputer = pickle.load(f)

apy_repository = get_apy_repository()

print("Model expects features:", model.n_features_in_)

OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"


class CropInput(BaseModel):
    N: float = Field(..., description="Nitrogen")
    P: float = Field(..., description="Phosphorus")
    K: float = Field(..., description="Potassium")
    temperature: float
    humidity: float
    ph: float
    rainfall: float


class NPKEstimator:
    """
    Data-driven fallback estimator for N/P/K using weather similarity
    from local training data. Advisory only; not a substitute for soil tests.
    """

    def __init__(self, csv_path: Path) -> None:
        self.ready = False
        self.error: str | None = None
        self.W: np.ndarray | None = None
        self.Y: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip().str.lower()
            need = [
                "temperature",
                "humidity",
                "rainfall",
                "nitrogen",
                "phosphorus",
                "potassium",
            ]
            for c in need:
                if c not in df.columns:
                    raise ValueError(f"Missing column in estimator dataset: {c}")
            df = df[need].dropna()
            self.W = df[["temperature", "humidity", "rainfall"]].to_numpy(float)
            self.Y = df[["nitrogen", "phosphorus", "potassium"]].to_numpy(float)
            self.mean = self.W.mean(axis=0)
            self.std = self.W.std(axis=0)
            self.std[self.std == 0] = 1.0
            if len(self.W) < 10:
                raise ValueError("Estimator dataset has too few rows.")
            self.ready = True
        except Exception as e:
            self.error = str(e)

    def estimate(
        self, temperature: float, humidity: float, rainfall: float, k: int = 40
    ) -> dict[str, Any]:
        if (
            not self.ready
            or self.W is None
            or self.Y is None
            or self.mean is None
            or self.std is None
        ):
            raise RuntimeError(self.error or "Estimator not ready.")

        x = np.array([temperature, humidity, rainfall], dtype=float)
        wz = (self.W - self.mean) / self.std
        xz = (x - self.mean) / self.std
        d = np.sqrt(((wz - xz) ** 2).sum(axis=1))
        idx = np.argsort(d)[: max(5, min(k, len(d)))]
        dsel = d[idx]
        ysel = self.Y[idx]

        w = 1.0 / (dsel + 1e-6)
        w = w / w.sum()
        est = (ysel * w[:, None]).sum(axis=0)

        spread = float(np.mean(np.std(ysel, axis=0)))
        mean_dist = float(np.mean(dsel))
        confidence = float(
            max(0.0, min(1.0, 1.0 / (1.0 + mean_dist + 0.08 * spread)))
        )

        return {
            "N": round(float(est[0]), 2),
            "P": round(float(est[1]), 2),
            "K": round(float(est[2]), 2),
            "neighbors_used": int(len(idx)),
            "confidence": round(confidence, 3),
        }


npk_estimator = NPKEstimator(ROOT / "cleaned_crop_data.csv")


def get_season(temp: float) -> str:
    if temp < 20:
        return "Winter"
    if temp < 30:
        return "Monsoon"
    return "Summer"


def _openweather_key() -> str:
    key = os.environ.get("OPENWEATHER_API_KEY", "").strip()
    if not key:
        raise HTTPException(
            status_code=503,
            detail="Weather is not configured. Set OPENWEATHER_API_KEY in a .env file.",
        )
    return key


def _http_get_json(url: str, timeout: int = 15) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "crop-edu-app/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "model_features": int(model.n_features_in_),
        "apy_csv_configured": apy_repository.is_loaded,
        "apy_note": apy_repository.load_error if not apy_repository.is_loaded else None,
        "npk_estimator_ready": npk_estimator.ready,
        "npk_estimator_note": None if npk_estimator.ready else npk_estimator.error,
    }


@app.get("/api/geocode/reverse")
def geocode_reverse(lat: float, lon: float) -> dict[str, Any]:
    """OSM Nominatim reverse lookup (low volume; respect usage policy)."""
    try:
        data = reverse_geocode_osm(lat, lon)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    addr = data.get("address") or {}
    district = addr.get("state_district") or addr.get("county") or ""
    return {
        "display_name": data.get("display_name"),
        "state": addr.get("state") or "",
        "district": district,
        "country_code": (addr.get("country_code") or "").upper(),
    }


@app.get("/api/yield-series")
def yield_series(
    crop: str,
    lat: float | None = None,
    lon: float | None = None,
    state: str | None = None,
    district: str | None = None,
) -> dict[str, Any]:
    """
    Time series from a **local** official APY CSV only (see data/apy/README.txt).
    If `state` + `district` are provided, reverse-geocoding is skipped.
    Otherwise, lat/lon are reverse-geocoded to state/district.
    """
    crop = (crop or "").strip().lower()
    if not crop:
        raise HTTPException(status_code=400, detail="crop is required.")

    if not apy_repository.is_loaded:
        return {
            "configured": False,
            "series": [],
            "message": apy_repository.load_error or "APY file not loaded.",
        }

    geo = None
    if state and district:
        state = state.strip()
        district = district.strip()
        if not state or not district:
            state = ""
            district = ""
    else:
        if lat is None or lon is None:
            return {
                "configured": True,
                "series": [],
                "message": "Provide either (state,district) or (lat,lon) for yield lookup.",
                "location": {"state": "", "district": ""},
            }
        try:
            geo = reverse_geocode_osm(lat, lon)
        except Exception as e:
            return {
                "configured": True,
                "series": [],
                "message": f"Geocoding failed: {e}. Please enter state and district manually.",
                "location": {"state": "", "district": ""},
            }

        addr = geo.get("address") or {}
        state = addr.get("state") or ""
        district = addr.get("state_district") or addr.get("county") or ""
        country = (addr.get("country_code") or "").upper()

        if country and country != "IN":
            return {
                "configured": True,
                "series": [],
                "message": "Yield lookup is intended for India; geocoded country is not IN.",
                "location": {"state": state, "district": district},
            }

        if not state or not district:
            return {
                "configured": True,
                "series": [],
                "message": "Geocoder did not return state/district needed to match APY rows. Please enter state and district manually.",
                "location": {"state": state, "district": district},
            }

    series, note = apy_repository.series_for(state, district, crop)
    return {
        "configured": True,
        "series": series,
        "note": note,
        "location": {
            "state": state,
            "district": district,
            "display_name": geo.get("display_name") if geo else None,
        },
    }


@app.get("/api/apy-districts")
def apy_districts(state: str, limit: int = Query(default=25, ge=1, le=200)) -> dict[str, Any]:
    """
    District suggestions extracted directly from your local APY files.
    Used to help users enter the exact district spelling present in the exports.
    """
    if not apy_repository.is_loaded:
        return {
            "configured": False,
            "message": apy_repository.load_error or "APY file not loaded.",
            "districts": [],
        }

    if hasattr(apy_repository, "districts_for_state"):
        districts = apy_repository.districts_for_state(state, limit=limit)
        return {"configured": True, "state": state, "districts": districts}

    return {
        "configured": False,
        "message": "District suggestions are not available for the current APY format.",
        "districts": [],
    }


@app.get("/api/weather")
def weather_by_coordinates(lat: float, lon: float) -> dict[str, Any]:
    key = _openweather_key()
    q = urllib.parse.urlencode(
        {"lat": lat, "lon": lon, "appid": key, "units": "metric"}
    )
    url = f"{OPENWEATHER_URL}?{q}"
    try:
        data = _http_get_json(url)
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=e.code, detail="OpenWeather returned an error.") from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return _normalize_weather_payload(data)


@app.get("/api/weather/city")
def weather_by_city(city: str) -> dict[str, Any]:
    if not city or not city.strip():
        raise HTTPException(status_code=400, detail="City name is required.")
    key = _openweather_key()
    q = urllib.parse.urlencode(
        {"q": city.strip(), "appid": key, "units": "metric"}
    )
    url = f"{OPENWEATHER_URL}?{q}"
    try:
        data = _http_get_json(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise HTTPException(status_code=404, detail="City not found.") from e
        raise HTTPException(status_code=e.code, detail="OpenWeather returned an error.") from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return _normalize_weather_payload(data)


@app.get("/api/estimate-npk")
def estimate_npk(temperature: float, humidity: float, rainfall: float) -> dict[str, Any]:
    if not npk_estimator.ready:
        raise HTTPException(
            status_code=503,
            detail=f"NPK estimator unavailable: {npk_estimator.error or 'initialization failed'}",
        )
    out = npk_estimator.estimate(temperature, humidity, rainfall)
    return {
        "estimated_npk": {"N": out["N"], "P": out["P"], "K": out["K"]},
        "confidence": out["confidence"],
        "neighbors_used": out["neighbors_used"],
        "note": (
            "Estimated from weather similarity using local training data. "
            "Use soil-test values when available."
        ),
    }


def _normalize_weather_payload(data: dict[str, Any]) -> dict[str, Any]:
    main = data.get("main") or {}
    rain = data.get("rain") or {}
    rain_1h = rain.get("1h")
    rain_3h = rain.get("3h")
    if rain_1h is not None:
        rainfall_mm = float(rain_1h)
        rainfall_note = "Rain in the last 1 hour (mm). Use manual override for seasonal/annual need."
    elif rain_3h is not None:
        rainfall_mm = float(rain_3h) / 3.0
        rainfall_note = "Approx. from last 3h rain (mm/h). Override if your model needs seasonal rainfall."
    else:
        rainfall_mm = 0.0
        rainfall_note = (
            "No recent rain in the API response. Enter rainfall manually for crop models "
            "that need seasonal or long-term precipitation."
        )

    return {
        "temperature_c": round(float(main.get("temp", 0)), 2),
        "humidity_pct": round(float(main.get("humidity", 0)), 2),
        "rainfall_mm": round(rainfall_mm, 2),
        "rainfall_note": rainfall_note,
        "location_name": data.get("name"),
        "country": (data.get("sys") or {}).get("country"),
    }


@app.post("/predict")
def predict(
    data: CropInput,
    top_k: int | None = Query(
        default=None,
        ge=1,
        description="Limit number of ranked crops returned (default: all classes).",
    ),
) -> dict[str, Any]:
    features = feature_matrix_from_inputs(
        data.N,
        data.P,
        data.K,
        data.temperature,
        data.humidity,
        data.ph,
        data.rainfall,
    )

    features = imputer.transform(features)
    features = scaler.transform(features)

    probs = model.predict_proba(features)[0]
    classes = model.classes_
    order = np.argsort(probs)[::-1]
    if top_k is not None:
        order = order[:top_k]

    rankings: list[dict[str, Any]] = []
    for rank_idx, ci in enumerate(order, start=1):
        p = float(probs[ci])
        rankings.append(
            {
                "rank": rank_idx,
                "crop": str(classes[ci]),
                "probability": round(p * 100, 2),
            }
        )

    top3_idx = probs.argsort()[-3:][::-1]
    top_probs = probs[top3_idx]
    s = top_probs.sum()
    top_probs_norm = top_probs / s if s > 0 else top_probs
    top_predictions = [
        {
            "crop": str(classes[top3_idx[i]]),
            "confidence": round(float(top_probs_norm[i]) * 100, 2),
        }
        for i in range(len(top3_idx))
    ]

    return {
        "season": get_season(data.temperature),
        "rankings": rankings,
        "top_predictions": top_predictions,
    }


app.mount("/", StaticFiles(directory=str(ROOT / "static"), html=True), name="static")
