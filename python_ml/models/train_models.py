"""
train_models.py
Master training script — generates data, trains all three models,
then persists every artifact.  Run this once to bootstrap the system.

Usage:
    python python_ml/models/train_models.py
"""

import os
import sys
import time

# ensure project root is on the path
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from data.generate_data import generate_dataset
from models.rain_prediction_model import train_rain_model
from models.drought_risk_model import train_drought_model
from models.crop_health_model import train_crop_model

DATA_PATH = os.path.join(ROOT, "data", "rwanda_agri_climate.csv")


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  AgriShield AI — Model Training Pipeline             ║")
    print("║  Food Insecurity & Climate Vulnerability System      ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    t0 = time.time()

    # ── Step 1: data ──────────────────────────────────────────────────────────
    print("► Step 1/4: Generating synthetic Rwanda agricultural dataset …")
    df = generate_dataset(n_days=1095, output_path=DATA_PATH)
    print(f"  {len(df):,} records ready.\n")

    # ── Step 2: rain model ────────────────────────────────────────────────────
    print("► Step 2/4: Training Rain Prediction models …")
    rain_rf, rain_mlp, rain_meta = train_rain_model(df)

    # ── Step 3: drought model ─────────────────────────────────────────────────
    print("\n► Step 3/4: Training Drought Risk models …")
    drought_gb, drought_lstm, drought_meta = train_drought_model(df)

    # ── Step 4: crop health model ─────────────────────────────────────────────
    print("\n► Step 4/4: Training Crop Health & Yield DNN …")
    crop_trainer, crop_meta = train_crop_model(df)

    elapsed = time.time() - t0
    print(f"\n✓ All models trained and saved in {elapsed:.1f}s")
    print("  Artifacts directory:", os.path.abspath(os.path.join(ROOT, "artifacts")))


if __name__ == "__main__":
    main()
