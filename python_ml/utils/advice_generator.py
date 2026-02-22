"""
advice_generator.py
Rule-based + ML-driven advice engine.

Takes predictions from all three models and converts them into
concrete, Kinyarwanda-context-aware actionable advice for farmers.
Also hosts the result-tracking / learning feedback loop.
"""

import os
import json
import datetime
import numpy as np
from pathlib import Path

FEEDBACK_LOG = os.path.join(
    os.path.dirname(__file__), "..", "data", "feedback_log.jsonl"
)
os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)


# ── Advice rules ──────────────────────────────────────────────────────────────

def _rain_advice(rain_result: dict) -> list[str]:
    tips = []
    prob = rain_result.get("probability", 0)
    if rain_result.get("rain_tomorrow"):
        tips.append(f"Rain expected tomorrow ({prob*100:.0f}% confidence). "
                    "Avoid irrigation today — save water and energy.")
        tips.append("Delay pesticide/fertilizer application until after rain.")
        if prob > 0.80:
            tips.append("High flood risk. Clear drainage channels around fields.")
    else:
        tips.append(f"No rain forecast ({(1-prob)*100:.0f}% confidence). "
                    "Schedule irrigation for tomorrow morning (cooler evaporation).")
        tips.append("Monitor soil moisture levels and top up if below 30%.")
    return tips


def _drought_advice(drought_result: dict) -> list[str]:
    tips = []
    level = drought_result.get("level", "none")
    score = drought_result.get("score", 0)

    if level == "none":
        tips.append("Soil moisture is adequate. Maintain current irrigation schedule.")
    elif level == "mild":
        tips.append("Mild water stress detected. Increase irrigation frequency by 20%.")
        tips.append("Apply mulch around crops to reduce soil evaporation.")
    elif level == "moderate":
        tips.append("Moderate drought (index: {:.2f}). Activate drip irrigation if available.".format(score))
        tips.append("Prioritise water for high-value crops (beans, vegetables).")
        tips.append("Consider early harvest of near-mature crops to avoid losses.")
    elif level == "severe":
        tips.append("SEVERE drought alert! Reduce field area under cultivation temporarily.")
        tips.append("Contact local agricultural office for emergency water supply assistance.")
        tips.append("Apply drought-tolerant varieties in any new planting.")
    elif level == "extreme":
        tips.append("EXTREME DROUGHT EMERGENCY. Activate all water reservoirs immediately.")
        tips.append("Report crop loss to Rwanda Agriculture Board (RAB) for insurance eligibility.")
        tips.append("Switch to drought-resistant crops: sorghum or cassava.")
    return tips


def _crop_health_advice(health_score: float, yield_est: float,
                         pest_result: dict) -> list[str]:
    tips = []
    pest_level = pest_result.get("level", "low")

    # Crop health score
    if health_score >= 75:
        tips.append(f"Crop health is excellent ({health_score:.0f}/100). "
                    "Maintain current management practices.")
    elif health_score >= 55:
        tips.append(f"Crop health is moderate ({health_score:.0f}/100). "
                    "Review fertilization schedule — apply nitrogen-rich fertilizer.")
    elif health_score >= 35:
        tips.append(f"Crop health is poor ({health_score:.0f}/100). "
                    "Urgent: check for nutrient deficiencies and soil pH.")
        tips.append("Consider foliar spray with micronutrients (zinc, iron).")
    else:
        tips.append(f"Critical crop health ({health_score:.0f}/100). "
                    "Immediate soil test and agronomist consultation required.")

    # Yield estimate
    tips.append(f"Estimated yield: {yield_est:.0f} kg/ha. "
                + ("On track for a good harvest." if yield_est >= 1500
                   else "Below average — intervention needed to improve output."))

    # Pest advice
    if pest_level in ("medium", "high", "critical"):
        tips.append(pest_result["message"])
        if pest_level == "medium":
            tips.append("Introduce natural predators (ladybirds, parasitic wasps) "
                        "as biological control.")
        elif pest_level in ("high", "critical"):
            tips.append("Apply recommended pesticide: consult RAB extension officer "
                        "for approved products and dosage rates.")
    return tips


def _general_season_advice(season: str, crop_type: str) -> list[str]:
    advice_map = {
        "rainy_A": [
            "Season A (March–May): Ideal planting window open.",
            f"Best time to plant {crop_type}. Ensure seeds are treated before sowing.",
        ],
        "rainy_B": [
            "Season B (September–November): Second planting opportunity.",
            f"Plant short-cycle {crop_type} varieties to maximise yield before dry season.",
        ],
        "dry": [
            "Dry season: Focus on soil preparation and irrigation infrastructure.",
            "Use organic compost to improve soil water retention for next season.",
        ],
    }
    return advice_map.get(season, [])


# ── Main advice generator ─────────────────────────────────────────────────────

def generate_advice(
    record: dict,
    rain_result: dict,
    drought_result: dict,
    health_score: float,
    yield_est: float,
    pest_result: dict,
) -> dict:
    """Aggregate all model outputs into a single comprehensive advice package."""

    advice = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "location": {
            "region": record.get("region", "Unknown"),
            "crop_type": record.get("crop_type", "Unknown"),
            "season": record.get("season", "Unknown"),
        },
        "predictions": {
            "rain": rain_result,
            "drought": drought_result,
            "crop_health": {
                "score": round(health_score, 2),
                "yield_estimate_kg_ha": round(yield_est, 2),
            },
            "pest": pest_result,
        },
        "advice": {
            "rain_management": _rain_advice(rain_result),
            "drought_management": _drought_advice(drought_result),
            "crop_management": _crop_health_advice(health_score, yield_est, pest_result),
            "seasonal_tips": _general_season_advice(
                record.get("season", "dry"),
                record.get("crop_type", "crop")
            ),
        },
        "priority_action": _priority_action(drought_result, rain_result, health_score),
        "feedback_id": _generate_feedback_id(),
    }

    return advice


def _priority_action(drought_result: dict, rain_result: dict, health: float) -> str:
    """Return the single most urgent action."""
    drought_level = drought_result.get("level", "none")
    if drought_level in ("extreme", "severe"):
        return "URGENT: Address water stress immediately — drought emergency declared."
    if health < 35:
        return "URGENT: Crop health critical — consult agronomist today."
    if rain_result.get("probability", 0) > 0.85:
        return "Rain imminent — secure drying areas and postpone field operations."
    if drought_level == "moderate":
        return "Activate supplemental irrigation — moderate drought developing."
    return "Continue regular monitoring — conditions are manageable."


def _generate_feedback_id() -> str:
    import uuid
    return str(uuid.uuid4())[:8].upper()


# ── Feedback / result tracking ────────────────────────────────────────────────

def log_farmer_feedback(feedback_id: str, actual_outcome: dict):
    """
    Track real-world farmer outcomes against predictions.
    Stored as JSONL for incremental retraining signals.

    actual_outcome example:
        {
          "rained": true,
          "actual_yield_kg_ha": 1200,
          "crop_health_observed": 60,
          "helpful_advice": ["drought_management"],
        }
    """
    entry = {
        "feedback_id": feedback_id,
        "logged_at": datetime.datetime.utcnow().isoformat() + "Z",
        "actual_outcome": actual_outcome,
    }
    with open(FEEDBACK_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[Feedback] Logged outcome for feedback_id={feedback_id}")
    return entry


def load_feedback_log() -> list[dict]:
    if not os.path.exists(FEEDBACK_LOG):
        return []
    with open(FEEDBACK_LOG, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_feedback_summary() -> dict:
    """Summarise tracked predictions vs. actuals — used for model drift monitoring."""
    logs = load_feedback_log()
    if not logs:
        return {"message": "No feedback data yet.", "count": 0}

    rain_correct = 0
    yield_errors = []
    for entry in logs:
        outcome = entry.get("actual_outcome", {})
        if "rained" in outcome and "predicted_rain" in outcome:
            rain_correct += int(outcome["rained"] == outcome["predicted_rain"])
        if "actual_yield_kg_ha" in outcome and "predicted_yield_kg_ha" in outcome:
            yield_errors.append(
                abs(outcome["actual_yield_kg_ha"] - outcome["predicted_yield_kg_ha"])
            )

    return {
        "count": len(logs),
        "rain_accuracy": round(rain_correct / len(logs), 4) if logs else None,
        "mean_yield_error_kg_ha": round(float(np.mean(yield_errors)), 2) if yield_errors else None,
    }
