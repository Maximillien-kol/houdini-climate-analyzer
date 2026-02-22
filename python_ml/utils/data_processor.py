"""
data_processor.py
Shared preprocessing utilities used by every model in the pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ── categorical columns ──────────────────────────────────────────────────────
CAT_COLS = ["region", "season", "crop_type", "soil_type"]

# ── numeric feature columns shared across models ─────────────────────────────
NUMERIC_FEATURES = [
    "temperature_c", "humidity_pct", "rainfall_mm", "soil_moisture_pct",
    "wind_speed_kmh", "solar_radiation_wm2", "ndvi", "pest_pressure_index",
    "fertilizer_applied_kg_ha", "irrigation_applied_mm", "drought_index",
]


def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    print(f"[Processor] Loaded {len(df):,} rows from {csv_path}")
    return df


def encode_categoricals(df: pd.DataFrame, fit: bool = True,
                         encoders: dict = None) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical columns; return (df, encoders)."""
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in CAT_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))

    return df, encoders


def build_feature_matrix(df: pd.DataFrame,
                          extra_cols: list[str] | None = None) -> np.ndarray:
    """Combine numeric + label-encoded categoricals into X matrix."""
    cols = NUMERIC_FEATURES + CAT_COLS
    if extra_cols:
        cols = cols + extra_cols
    cols = [c for c in cols if c in df.columns]
    return df[cols].values.astype(np.float32)


def scale_features(X_train: np.ndarray, X_test: np.ndarray,
                   fit: bool = True,
                   scaler: StandardScaler | None = None
                   ) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    if scaler is None:
        scaler = StandardScaler()
    if fit:
        X_train = scaler.fit_transform(X_train)
    else:
        X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, scaler


def prepare_split(df: pd.DataFrame, target_col: str,
                  test_size: float = 0.2,
                  extra_cols: list[str] | None = None
                  ) -> tuple:
    """Full encode → split → scale pipeline. Returns (X_tr, X_te, y_tr, y_te, meta)."""
    df, encoders = encode_categoricals(df, fit=True)
    X = build_feature_matrix(df, extra_cols)
    y = df[target_col].values.astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    X_tr, X_te, scaler = scale_features(X_tr, X_te)

    meta = {"encoders": encoders, "scaler": scaler}
    return X_tr, X_te, y_tr, y_te, meta


def save_artifacts(name: str, meta: dict):
    path = os.path.join(ARTIFACTS_DIR, f"{name}_meta.pkl")
    joblib.dump(meta, path)
    print(f"[Processor] Saved artifacts → {path}")


def load_artifacts(name: str) -> dict:
    path = os.path.join(ARTIFACTS_DIR, f"{name}_meta.pkl")
    return joblib.load(path)


def preprocess_single(record: dict, meta: dict,
                       extra_cols: list[str] | None = None) -> np.ndarray:
    """Turn a single incoming dict into a scaled feature vector."""
    df = pd.DataFrame([record])
    df, _ = encode_categoricals(df, fit=False, encoders=meta["encoders"])
    X = build_feature_matrix(df, extra_cols)
    X = meta["scaler"].transform(X)
    return X.astype(np.float32)
