"""
prediction_service.py
Flask REST API — exposes all three AI models as HTTP endpoints.
Node.js backend calls these endpoints to serve farmers.

Endpoints:
  POST /predict/rain          → rain_tomorrow prediction
  POST /predict/drought       → drought index + risk level
  POST /predict/crop          → crop health score + yield estimate
  POST /predict/full          → all three + aggregated advice (manual data)
  GET  /predict/realtime      → full report using REAL Open-Meteo weather
  GET  /predict/realtime/<region> → realtime for specific Rwanda region
  POST /feedback              → log real-world outcomes
  GET  /feedback/summary      → feedback / drift summary
  GET  /health                → service health check
"""

import os
import sys
import json
import logging

# Load .env before anything else
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from flask import Flask, request, jsonify
import numpy as np

# Models
from models.rain_prediction_model import (
    RainRandomForest, RainMLPTrainer, ensemble_predict as rain_ensemble
)
from models.drought_risk_model import (
    DroughtGBRegressor, DroughtLSTMTrainer, classify_drought
)
from models.crop_health_model import CropHealthTrainer, classify_pest

# Utils
from utils.data_processor import load_artifacts, preprocess_single
from utils.advice_generator import (
    generate_advice, log_farmer_feedback, compute_feedback_summary
)
from utils.weather_fetcher import fetch_current_weather
from utils.groq_advisor import generate_groq_advice, enhance_advice_with_groq

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Lazy model registry ───────────────────────────────────────────────────────
MODELS = {}


def _load_models():
    global MODELS
    if MODELS:
        return

    log.info("Loading model artifacts …")
    try:
        # Rain
        rain_meta = load_artifacts("rain_prediction")
        rain_rf = RainRandomForest(); rain_rf.load()
        rain_mlp = RainMLPTrainer(input_dim=15); rain_mlp.load(input_dim=15)

        # Drought
        drought_meta = load_artifacts("drought_risk")
        drought_gb = DroughtGBRegressor(); drought_gb.load()

        # Crop
        crop_meta = load_artifacts("crop_health")
        crop_trainer = CropHealthTrainer(input_dim=15); crop_trainer.load()

        MODELS = {
            "rain":   {"rf": rain_rf,   "mlp": rain_mlp,      "meta": rain_meta},
            "drought":{"gb": drought_gb,                       "meta": drought_meta},
            "crop":   {"trainer": crop_trainer,                "meta": crop_meta},
        }
        log.info("All models loaded successfully.")
    except Exception as e:
        log.error(f"Model loading failed: {e}")
        log.warning("Running in DEMO mode — returning mock predictions.")
        MODELS = {"demo": True}


# ── Request parsing ───────────────────────────────────────────────────────────

def _parse_record() -> dict:
    """Extract and validate JSON body."""
    data = request.get_json(silent=True)
    if not data:
        raise ValueError("Request body must be valid JSON.")
    return data


def _safe_preprocess(record: dict, model_name: str) -> np.ndarray:
    meta = MODELS[model_name]["meta"]
    return preprocess_single(record, meta)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    _load_models()
    return jsonify({
        "status": "ok",
        "models_loaded": "demo" not in MODELS,
        "service": "AgriShield AI Prediction Service",
        "version": "1.0.0",
    })


@app.route("/predict/rain", methods=["POST"])
def predict_rain():
    _load_models()
    try:
        record = _parse_record()
        if "demo" in MODELS:
            return jsonify(_mock_rain())
        X = _safe_preprocess(record, "rain")
        result = rain_ensemble(MODELS["rain"]["rf"], MODELS["rain"]["mlp"], X)
        return jsonify({"success": True, "prediction": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/predict/drought", methods=["POST"])
def predict_drought():
    _load_models()
    try:
        record = _parse_record()
        if "demo" in MODELS:
            return jsonify(_mock_drought())
        X = _safe_preprocess(record, "drought")
        score = float(MODELS["drought"]["gb"].predict(X)[0])
        result = classify_drought(score)
        return jsonify({"success": True, "prediction": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/predict/crop", methods=["POST"])
def predict_crop():
    _load_models()
    try:
        record = _parse_record()
        if "demo" in MODELS:
            return jsonify(_mock_crop())
        X = _safe_preprocess(record, "crop")
        h, y = MODELS["crop"]["trainer"].predict(X)
        pest_level = classify_pest(float(record.get("pest_pressure_index", 0.2)))
        return jsonify({
            "success": True,
            "prediction": {
                "crop_health_score": round(float(h[0]), 2),
                "yield_estimate_kg_ha": round(float(y[0]), 2),
                "pest": pest_level,
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/predict/full", methods=["POST"])
def predict_full():
    """
    Full pipeline: rain + drought + crop → aggregated advice + Groq enhancement.
    This is the primary endpoint called by Node.js.
    """
    _load_models()
    try:
        record = _parse_record()

        if "demo" in MODELS:
            rain_res   = _mock_rain()["prediction"]
            drought_res= _mock_drought()["prediction"]
            crop_res   = _mock_crop()["prediction"]
            health     = crop_res["crop_health_score"]
            yield_est  = crop_res["yield_estimate_kg_ha"]
            pest_res   = crop_res["pest"]
        else:
            # Rain
            X_rain = _safe_preprocess(record, "rain")
            rain_res = rain_ensemble(MODELS["rain"]["rf"], MODELS["rain"]["mlp"], X_rain)

            # Drought
            X_drought = _safe_preprocess(record, "drought")
            d_score = float(MODELS["drought"]["gb"].predict(X_drought)[0])
            drought_res = classify_drought(d_score)

            # Crop
            X_crop = _safe_preprocess(record, "crop")
            h, y = MODELS["crop"]["trainer"].predict(X_crop)
            health    = float(h[0])
            yield_est = float(y[0])
            pest_res  = classify_pest(float(record.get("pest_pressure_index", 0.2)))

        # Rule-based advice
        rule_advice = generate_advice(record, rain_res, drought_res,
                                      health, yield_est, pest_res)

        # Groq LLM enhancement
        groq_advice = generate_groq_advice(record, rain_res, drought_res,
                                           health, yield_est, pest_res)
        final = enhance_advice_with_groq(rule_advice, groq_advice)

        return jsonify({"success": True, "report": final})

    except Exception as e:
        log.exception("Error in /predict/full")
        return jsonify({"success": False, "error": str(e)}), 500


# ── Real-time endpoint (Open-Meteo + ML + Groq) ───────────────────────────────

def _run_realtime_pipeline(region: str, crop_type: str,
                            fertilizer: float, irrigation: float):
    """Shared logic for both realtime endpoints."""
    _load_models()

    # 1 — Fetch real weather from Open-Meteo
    record = fetch_current_weather(
        region=region,
        crop_type=crop_type,
        fertilizer_kg_ha=fertilizer,
        irrigation_mm=irrigation,
    )

    if "demo" in MODELS:
        rain_res   = _mock_rain()["prediction"]
        drought_res= _mock_drought()["prediction"]
        crop_res   = _mock_crop()["prediction"]
        health     = crop_res["crop_health_score"]
        yield_est  = crop_res["yield_estimate_kg_ha"]
        pest_res   = crop_res["pest"]
    else:
        # Strip private metadata keys before preprocessing
        clean = {k: v for k, v in record.items() if not k.startswith("_")}

        X_rain    = _safe_preprocess(clean, "rain")
        rain_res  = rain_ensemble(MODELS["rain"]["rf"], MODELS["rain"]["mlp"], X_rain)

        X_drought = _safe_preprocess(clean, "drought")
        d_score   = float(MODELS["drought"]["gb"].predict(X_drought)[0])
        drought_res = classify_drought(d_score)

        X_crop    = _safe_preprocess(clean, "crop")
        h, y      = MODELS["crop"]["trainer"].predict(X_crop)
        health    = float(h[0])
        yield_est = float(y[0])
        pest_res  = classify_pest(float(record.get("pest_pressure_index", 0.2)))

    # Rule-based advice
    rule_advice = generate_advice(record, rain_res, drought_res,
                                  health, yield_est, pest_res)

    # Groq LLM enhancement (real weather → richer context for LLM)
    groq_advice = generate_groq_advice(record, rain_res, drought_res,
                                       health, yield_est, pest_res)
    final = enhance_advice_with_groq(rule_advice, groq_advice)
    final["weather_data"] = record

    return final


@app.route("/predict/realtime", methods=["GET"])
def predict_realtime():
    """
    Fetch REAL weather for Northern Rwanda (Musanze) and run full AI pipeline.
    Query params: region, crop_type, fertilizer_kg_ha, irrigation_mm
    """
    try:
        region      = request.args.get("region", "Northern")
        crop_type   = request.args.get("crop_type", "maize")
        fertilizer  = float(request.args.get("fertilizer_kg_ha", 50))
        irrigation  = float(request.args.get("irrigation_mm", 0))

        final = _run_realtime_pipeline(region, crop_type, fertilizer, irrigation)
        return jsonify({"success": True, "report": final})

    except Exception as e:
        log.exception("Error in /predict/realtime")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/predict/realtime/<region>", methods=["GET"])
def predict_realtime_region(region):
    """Shorthand: GET /predict/realtime/Northern?crop_type=maize"""
    try:
        crop_type  = request.args.get("crop_type", "maize")
        fertilizer = float(request.args.get("fertilizer_kg_ha", 50))
        irrigation = float(request.args.get("irrigation_mm", 0))

        final = _run_realtime_pipeline(region, crop_type, fertilizer, irrigation)
        return jsonify({"success": True, "report": final})

    except Exception as e:
        log.exception(f"Error in /predict/realtime/{region}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── GET /forecast ────────────────────────────────────────────────────────────
@app.route("/forecast", methods=["GET"])
@app.route("/forecast/<region>", methods=["GET"])
def climate_forecast(region: str = None):
    """
    Run a 5-year Holt-Winters climate forecast from 10 years of real Open-Meteo data.
    Query params:
      region     (str)  : Rwanda province — Northern/Eastern/Southern/Western/Kigali
      start_year (int)  : first year of historical window (default: 2015)
    Returns:
      JSON with monthly forecasts 2026-2030 + summary statistics.
    """
    import warnings
    warnings.filterwarnings("ignore")
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from forecast import (
            fetch_historical, train_and_forecast,
            MONTHLY_RAIN_NORMALS, drought_style, PROVINCE_COORDS
        )
        import numpy as np, pandas as pd

        region     = region or request.args.get("region", "Northern")
        start_year = int(request.args.get("start_year", 2015))

        if region not in PROVINCE_COORDS:
            return jsonify({"success": False,
                            "error": f"Unknown region '{region}'. Valid: {list(PROVINCE_COORDS.keys())}"}), 400

        # ── Historical data ────────────────────────────────────────────────────
        hist_df = fetch_historical(region=region, start_year=start_year, end_year=start_year + 9)

        # ── Train + forecast ───────────────────────────────────────────────────
        rain_fc, rain_lo, rain_hi   = train_and_forecast(hist_df["rain_sum"], periods=60)
        temp_fc, temp_lo, temp_hi   = train_and_forecast(hist_df["temp_mean"], periods=60)
        rain_fc = np.clip(rain_fc, 0, None)
        rain_lo = np.clip(rain_lo, 0, None)
        rain_hi = np.clip(rain_hi, 0, None)

        fc_dates = pd.date_range("2026-02-01", periods=60, freq="MS")

        # ── Build month-by-month records ───────────────────────────────────────
        monthly = []
        for i, dt in enumerate(fc_dates):
            normal  = MONTHLY_RAIN_NORMALS[dt.month]
            drought = float(max(0, min(1, 1.0 - rain_fc[i] / (normal + 1))))
            d_lo    = float(max(0, min(1, 1.0 - rain_hi[i] / (normal + 1))))
            d_hi    = float(max(0, min(1, 1.0 - rain_lo[i] / (normal + 1))))
            _, _, d_label = drought_style(drought)
            monthly.append({
                "date":         dt.strftime("%Y-%m"),
                "year":         dt.year,
                "month":        dt.month,
                "rain_mm":      round(float(rain_fc[i]), 1),
                "rain_lower":   round(float(rain_lo[i]), 1),
                "rain_upper":   round(float(rain_hi[i]), 1),
                "temp_c":       round(float(temp_fc[i]), 1),
                "temp_lower":   round(float(temp_lo[i]), 1),
                "temp_upper":   round(float(temp_hi[i]), 1),
                "drought_index":round(drought, 4),
                "drought_lower":round(d_lo, 4),
                "drought_upper":round(d_hi, 4),
                "drought_level":d_label,
            })

        # ── Year summaries ─────────────────────────────────────────────────────
        fc_df  = pd.DataFrame(monthly)
        yearly = []
        for yr in sorted(fc_df["year"].unique()):
            yr_df   = fc_df[fc_df["year"] == yr]
            avg_d   = float(yr_df["drought_index"].mean())
            _, _, lbl = drought_style(avg_d)
            worst   = yr_df.loc[yr_df["drought_index"].idxmax()]
            yearly.append({
                "year":           int(yr),
                "total_rain_mm":  round(float(yr_df["rain_mm"].sum()), 0),
                "avg_temp_c":     round(float(yr_df["temp_c"].mean()), 1),
                "avg_drought":    round(avg_d, 4),
                "drought_level":  lbl,
                "worst_month":    worst["date"],
                "extreme_months": int((yr_df["drought_index"] >= 0.8).sum()),
                "severe_months":  int((yr_df["drought_index"] >= 0.6).sum()),
            })

        # ── Historical stats ───────────────────────────────────────────────────
        hist_rain_annual = float(
            hist_df.groupby(hist_df["date"].dt.year)["rain_sum"].sum().mean()
        )
        hist_drought_avg = float(hist_df["drought"].mean())
        fc_drought_avg   = float(fc_df["drought_index"].mean())
        trend = ("worsening" if fc_drought_avg > hist_drought_avg + 0.05
                 else "improving" if fc_drought_avg < hist_drought_avg - 0.05
                 else "stable")

        coords = PROVINCE_COORDS[region]
        return jsonify({
            "success": True,
            "region":  region,
            "location": {
                "province": region,
                "city":     coords["city"],
                "lat":      coords["lat"],
                "lon":      coords["lon"],
            },
            "historical": {
                "years":             f"{start_year}–{start_year + 9}",
                "months_used":       len(hist_df),
                "avg_annual_rain_mm":round(hist_rain_annual, 0),
                "avg_temp_c":        round(float(hist_df["temp_mean"].mean()), 1),
                "avg_drought_index": round(hist_drought_avg, 4),
            },
            "forecast": {
                "horizon":           "2026-02 to 2030-12 (60 months)",
                "drought_trend":     trend,
                "avg_annual_rain_mm":round(float(fc_df["rain_mm"].mean() * 12), 0),
                "avg_drought_index": round(fc_drought_avg, 4),
                "extreme_months":    int((fc_df["drought_index"] >= 0.8).sum()),
                "severe_months":     int((fc_df["drought_index"] >= 0.6).sum()),
                "model":             "Holt-Winters ExponentialSmoothing (seasonal=12, trend=add)",
            },
            "monthly_forecast":  monthly,
            "yearly_forecast":   yearly,
        })

    except Exception as e:
        log.exception("/forecast error")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        data = _parse_record()
        feedback_id    = data.get("feedback_id", "UNKNOWN")
        actual_outcome = data.get("actual_outcome", {})
        entry = log_farmer_feedback(feedback_id, actual_outcome)
        return jsonify({"success": True, "logged": entry})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/feedback/summary", methods=["GET"])
def feedback_summary():
    summary = compute_feedback_summary()
    return jsonify({"success": True, "summary": summary})


# ── Mock responses (demo / fallback) ─────────────────────────────────────────

def _mock_rain():
    return {"prediction": {"rain_tomorrow": True, "probability": 0.72,
                           "rf_probability": 0.70, "mlp_probability": 0.75}}

def _mock_drought():
    return {"prediction": {"level": "mild", "score": 0.30,
                           "message": "Mild dryness — monitor soil moisture closely."}}

def _mock_crop():
    return {"prediction": {
        "crop_health_score": 68.5,
        "yield_estimate_kg_ha": 1400.0,
        "pest": {"level": "low", "pest_pressure_index": 0.18,
                 "message": "Pest pressure is low — routine monitoring sufficient."}
    }}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _load_models()
    port = int(os.environ.get("FLASK_PORT", 5001))
    log.info(f"AgriShield AI Service starting on port {port} …")
    app.run(host="0.0.0.0", port=port, debug=False)
