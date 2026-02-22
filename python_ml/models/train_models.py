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
import logging

# ensure project root is on the path
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from data.generate_data import generate_dataset
from data.document_integrator import load_merged_dataset, DOCUMENTS_DIR
from models.rain_prediction_model import train_rain_model
from models.drought_risk_model import train_drought_model
from models.crop_health_model import train_crop_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

DATA_PATH = os.path.join(ROOT, "data", "rwanda_agri_climate.csv")


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  AgriShield AI — Model Training Pipeline             ║")
    print("║  Food Insecurity & Climate Vulnerability System      ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    t0 = time.time()

    # ── Step 1: data ──────────────────────────────────────────────────────────
    print("► Step 1/4: Generating synthetic Rwanda agricultural dataset …")
    df_synthetic = generate_dataset(n_days=1095, output_path=DATA_PATH)
    print(f"  {len(df_synthetic):,} synthetic records generated.")

    # ── Step 1b: merge document data ──────────────────────────────────────────
    doc_folder = os.path.abspath(DOCUMENTS_DIR)
    has_docs = os.path.isdir(doc_folder) and any(
        f.lower().endswith((".pdf", ".docx", ".doc"))
        for _, _, files in os.walk(doc_folder)
        for f in files
    )
    if has_docs:
        print(f"  Documents found in {doc_folder}")
        print("  Parsing & integrating forecast documents …")
    else:
        print(f"  No PDF/DOCX documents found in {doc_folder}")
        print("  (Add Meteorwanda monthly forecasts there to enrich training data)")

    df = load_merged_dataset(
        synthetic_csv=DATA_PATH,
        documents_dir=doc_folder,
        save_merged=True,
    )
    n_doc = (df.get("source", "") == "document").sum() if "source" in df.columns else 0
    print(f"  Final training set: {len(df):,} rows  "
          f"({len(df_synthetic):,} synthetic + {n_doc:,} from documents)\n")

    # Training models only need the numeric/categorical columns — drop aux cols
    aux_cols = ["source", "data_weight"]
    df_train = df.drop(columns=[c for c in aux_cols if c in df.columns])

    # ── Step 2: rain model ────────────────────────────────────────────────────
    print("► Step 2/4: Training Rain Prediction models …")
    rain_rf, rain_mlp, rain_meta = train_rain_model(df_train)

    # ── Step 3: drought model ─────────────────────────────────────────────────
    print("\n► Step 3/4: Training Drought Risk models …")
    drought_gb, drought_lstm, drought_meta = train_drought_model(df_train)

    # ── Step 4: crop health model ─────────────────────────────────────────────
    print("\n► Step 4/4: Training Crop Health & Yield DNN …")
    crop_trainer, crop_meta = train_crop_model(df_train)

    elapsed = time.time() - t0
    print(f"\n✓ All models trained and saved in {elapsed:.1f}s")
    print("  Artifacts directory:", os.path.abspath(os.path.join(ROOT, "artifacts")))


if __name__ == "__main__":
    main()
