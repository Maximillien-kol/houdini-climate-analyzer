"""
main.py — AgriShield AI entry point.

Modes:
  train    : generate data and train all models
  serve    : start the Flask prediction API
  demo     : run a quick end-to-end demo prediction (no trained models needed)
  feedback : show feedback / drift summary

Usage:
  python main.py train
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


def run_serve():
    from api.prediction_service import app, _load_models
    import logging

    logging.basicConfig(level=logging.INFO)
    _load_models()

    port = int(os.environ.get("FLASK_PORT", 5001))
    print(f"\n► AgriShield AI Prediction Service running on http://0.0.0.0:{port}")
    print("  Endpoints: /health  /predict/rain  /predict/drought  /predict/crop  /predict/full")
    print("  Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=port, debug=False)


def run_demo():
    """
    End-to-end demo using mock data for a farmer in Eastern Rwanda.
    Does NOT require trained models — uses the demo fallback mode.
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
    print("║  AgriShield AI — DEMO Mode                           ║")
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
    "train": run_train,
    "serve": run_serve,
    "demo":  run_demo,
    "feedback": run_feedback,
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd not in COMMANDS:
        print(f"Unknown command '{cmd}'. Available: {list(COMMANDS.keys())}")
        sys.exit(1)
    COMMANDS[cmd]()
