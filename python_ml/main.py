"""
main.py - AgriShield AI entry point.

Modes:
  train    : generate synthetic data and train all models
  retrain  : parse documents + generate data + train all models  ← use this
  serve    : start the Flask prediction API
  demo     : run a quick end-to-end demo prediction (no trained models needed)
  feedback : show feedback / drift summary

Usage:
  python main.py retrain          # includes PDF/DOCX documents in training
  python main.py train            # synthetic data only
  python main.py serve
  python main.py demo
  python main.py feedback
"""

import sys
import os
import json

ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)


def run_train():
    from models.train_models import main as train_main
    train_main()


def run_retrain():
    """
    Full retrain: generates synthetic data, parses any PDF/DOCX documents
    in data/documents/, merges them, then trains all models.
    This is the recommended command after adding new forecast documents.
    """
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    from data.generate_data import generate_dataset
    from data.document_integrator import load_merged_dataset, DOCUMENTS_DIR
    from models.rain_prediction_model import train_rain_model
    from models.drought_risk_model import train_drought_model
    from models.crop_health_model import train_crop_model
    import time

    DATA_PATH = os.path.join(ROOT, "data", "rwanda_agri_climate.csv")

    print("╔══════════════════════════════════════════════════════╗")
    print("║  AgriShield AI - Full Retrain (Synthetic + Documents)║")
    print("╚══════════════════════════════════════════════════════╝\n")
    t0 = time.time()

    print("► Step 1/4: Generating synthetic dataset …")
    df_synthetic = generate_dataset(n_days=1095, output_path=DATA_PATH)
    print(f"  {len(df_synthetic):,} synthetic records generated.")

    doc_folder = os.path.abspath(DOCUMENTS_DIR)
    pdf_count = sum(
        1 for _, _, files in os.walk(doc_folder)
        for f in files if f.lower().endswith((".pdf", ".docx", ".doc"))
    )
    if pdf_count:
        print(f"  Found {pdf_count} document(s) in {doc_folder}")
        print("  Parsing & merging forecast documents …")
    else:
        print(f"  No documents found in {doc_folder} - using synthetic data only.")

    df = load_merged_dataset(synthetic_csv=DATA_PATH, documents_dir=doc_folder, save_merged=True)
    n_doc = int((df.get("source", "") == "document").sum()) if "source" in df.columns else 0
    print(f"  Training set: {len(df):,} rows ({len(df_synthetic):,} synthetic + {n_doc:,} from documents)\n")

    df_train = df.drop(columns=[c for c in ["source", "data_weight"] if c in df.columns])

    print("► Step 2/4: Training Rain Prediction models …")
    train_rain_model(df_train)

    print("\n► Step 3/4: Training Drought Risk models …")
    train_drought_model(df_train)

    print("\n► Step 4/4: Training Crop Health & Yield DNN …")
    train_crop_model(df_train)

    print(f"\n✓ Done in {time.time()-t0:.1f}s - models saved to artifacts/")


def run_serve():
    from api.prediction_service import app, _load_models
    import logging

    logging.basicConfig(level=logging.INFO)
    _load_models()

    port = int(os.environ.get("FLASK_PORT", 5001))
    print(f"\n► Rwac V.0.1 Prediction Service running on http://0.0.0.0:{port}")
    print("  Endpoints: /health  /predict/rain  /predict/drought  /predict/crop  /predict/full")
    print("  Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=port, debug=False)


def run_demo():
    """
    End-to-end demo using mock data for a farmer in Eastern Rwanda.
    Does NOT require trained models - uses the demo fallback mode.
    """
    import requests

    sample_record = {
        "region": "Eastern",
        "season": "rainy_A",
        "crop_type": "maize",
        "soil_type": "loam",
        "temperature_c": 22.5,
        "humidity_pct": 68.0,
        "rainfall_mm": 45.0,
        "soil_moisture_pct": 35.0,
        "wind_speed_kmh": 10.0,
        "solar_radiation_wm2": 210.0,
        "ndvi": 0.62,
        "pest_pressure_index": 0.28,
        "fertilizer_applied_kg_ha": 50,
        "irrigation_applied_mm": 0.0,
        "drought_index": 0.22,
    }

    port = int(os.environ.get("FLASK_PORT", 5001))
    url = f"http://localhost:{port}/predict/full"

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  AgriShield AI - DEMO Mode                           ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"\nSending sample record to {url} …")
    print(json.dumps(sample_record, indent=2))

    try:
        resp = requests.post(url, json=sample_record, timeout=10)
        report = resp.json()
        print("\n── AI Report ──────────────────────────────────────────")
        print(json.dumps(report, indent=2))
    except Exception as e:
        print(f"\n[Demo] Could not reach API: {e}")
        print("  → Start the service first with: python main.py serve")


def run_feedback():
    from utils.advice_generator import compute_feedback_summary, load_feedback_log

    summary = compute_feedback_summary()
    logs = load_feedback_log()

    print("\n── Feedback / Model Drift Summary ─────────────────────")
    print(json.dumps(summary, indent=2))
    print(f"\nTotal feedback entries: {len(logs)}")
    if logs:
        print("Last 3 entries:")
        for entry in logs[-3:]:
            print(" ", json.dumps(entry))


COMMANDS = {
    "train":    run_train,
    "retrain":  run_retrain,
    "serve":    run_serve,
    "demo":     run_demo,
    "feedback": run_feedback,
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd not in COMMANDS:
        print(f"Unknown command '{cmd}'. Available: {list(COMMANDS.keys())}")
        sys.exit(1)
    COMMANDS[cmd]()
