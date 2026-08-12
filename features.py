"""
Shared feature construction for training and /predict inference.

We keep all seven raw agronomic inputs. A small set of *derived* scalars
helps the model exploit structure in the data (correlated blocks) without
removing anything the farmer or API can measure.

See ENGINEERED_FEATURE_JUSTIFICATION for exam / report wording.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FEATURE_NAMES = [
    "nitrogen",
    "phosphorus",
    "potassium",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]

# Kept deliberately small: each name below must be defensible in writing.
# Dropped vs an earlier 14-feature draft:
# - npk_mean — perfectly redundant with npk_total after StandardScaler.
# - npk_cv — hard to explain to non-technical reviewers; ratio + total suffice.
# - ph_npk_interaction — pH stays as a raw input; interaction was optional complexity.
ENGINEERED_FEATURE_NAMES = [
    "npk_total",
    "n_to_pkratio",
    "temp_humidity",
    "rainfall_humidity",
]

ALL_FEATURE_NAMES = BASE_FEATURE_NAMES + ENGINEERED_FEATURE_NAMES

# For reports, vivas, and README: **what** the column is and **why** we use it.
BASE_FEATURE_JUSTIFICATION: dict[str, tuple[str, str]] = {
    "nitrogen": ("Soil nitrogen N", "Direct model input; primary macronutrient."),
    "phosphorus": ("Soil phosphorus P", "Direct model input; primary macronutrient."),
    "potassium": ("Soil potassium K", "Direct model input; primary macronutrient."),
    "temperature": ("Air / ambient temperature", "Direct climate driver for crop suitability."),
    "humidity": ("Relative humidity", "Moisture regime; correlates with P in this dataset."),
    "ph": ("Soil pH", "Acidity/alkalinity affects nutrient availability."),
    "rainfall": ("Rainfall (e.g. mm)", "Water supply; strongly correlates with K here."),
}

ENGINEERED_FEATURE_JUSTIFICATION: dict[str, tuple[str, str]] = {
    "npk_total": (
        "N + P + K",
        "One number for overall macronutrient ‘load’. We do not remove N, P, or K; "
        "this is an *extra* summary the forest can use like a fertiliser index.",
    ),
    "n_to_pkratio": (
        "N / (P + K + ε)",
        "Fertiliser *balance*: two samples can share similar N+P+K totals but differ "
        "in N vs P+K skew. Complements npk_total. Matches the idea that N and P are "
        "linked but not identical in the data.",
    ),
    "temp_humidity": (
        "(temperature × humidity) / 100",
        "Combines heat and air moisture—closer to how growers think about ‘stress’ than "
        "either variable alone. Supported by humidity appearing in several nutrient "
        "correlations in feature_analysis.",
    ),
    "rainfall_humidity": (
        "(rainfall × humidity) / 100",
        "Joins water supply (rain) with atmospheric moisture. The cleaned dataset shows "
        "strong K–rainfall and P–humidity ties; this term models *joint* wetness without "
        "dropping rainfall or humidity as raw features.",
    ),
}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expect columns: BASE_FEATURE_NAMES. Returns base + ENGINEERED columns."""
    missing = set(BASE_FEATURE_NAMES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing base columns: {sorted(missing)}")

    out = df[BASE_FEATURE_NAMES].copy()
    N = out["nitrogen"].astype(float)
    P = out["phosphorus"].astype(float)
    K = out["potassium"].astype(float)
    eps = 1e-6

    out["npk_total"] = N + P + K
    out["n_to_pkratio"] = N / (P + K + eps)
    out["temp_humidity"] = out["temperature"] * out["humidity"] / 100.0
    out["rainfall_humidity"] = out["rainfall"] * out["humidity"] / 100.0

    return out


def dataframe_from_api_row(
    N: float,
    P: float,
    K: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float,
) -> pd.DataFrame:
    """Single-row DataFrame with base columns only."""
    return pd.DataFrame(
        [
            {
                "nitrogen": N,
                "phosphorus": P,
                "potassium": K,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall,
            }
        ]
    )


def feature_matrix_from_inputs(
    N: float,
    P: float,
    K: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float,
) -> np.ndarray:
    """(1, n_features) float array in ALL_FEATURE_NAMES order."""
    base = dataframe_from_api_row(N, P, K, temperature, humidity, ph, rainfall)
    full = add_engineered_features(base)
    return full[ALL_FEATURE_NAMES].to_numpy(dtype=float)


def print_feature_justification() -> None:
    """Stdout text block for reports / slides."""
    print("=== Base features (7) — all measured / supplied ===\n")
    for name in BASE_FEATURE_NAMES:
        formula, why = BASE_FEATURE_JUSTIFICATION[name]
        print(f"  • {name}")
        print(f"      What: {formula}")
        print(f"      Why:  {why}\n")
    print("=== Engineered features (4) — derived only from the base seven ===\n")
    for name in ENGINEERED_FEATURE_NAMES:
        formula, why = ENGINEERED_FEATURE_JUSTIFICATION[name]
        print(f"  • {name}")
        print(f"      What: {formula}")
        print(f"      Why:  {why}\n")
