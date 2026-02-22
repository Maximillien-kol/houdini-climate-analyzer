"""
groq_advisor.py
Integrates Groq LLM (llama-3.3-70b-versatile) to generate high-quality,
context-aware, structured farming advice from model predictions + real weather.

Falls back to rule-based advice when Groq is unavailable / key missing.
"""

import os
import json
import logging

log = logging.getLogger(__name__)

# Load from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Groq client (lazy import) ─────────────────────────────────────────────────
_groq_client = None

def _get_client():
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set.")
    from groq import Groq
    _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are AgriShield, an expert agricultural AI advisor specialising in
Rwanda's smallholder farming systems. You have deep knowledge of:
- Rwanda's bimodal rainfall calendar (Season A: March-May, Season B: September-November)
- Key staple crops: maize, beans, sorghum, cassava, sweet potato
- Rwanda Agriculture Board (RAB) recommended practices
- Climate-smart agriculture techniques for East African highlands
- Food security risks and drought response strategies

You receive structured data from three AI models:
  1. Rain Prediction Model (Random Forest + MLP ensemble)
  2. Drought Risk Model (Gradient Boosting + LSTM)
  3. Crop Health & Yield Model (multi-output PyTorch DNN)

Your job: convert these ML outputs into CONCRETE, ACTIONABLE advice for the farmer.
Be specific, practical, and direct. Use simple language — assume the farmer has a
primary school education. Prioritize the most urgent action first.

ALWAYS respond with a valid JSON object — no extra text before or after — in this exact structure:
{
  "priority_action": "Single most urgent action in one sentence.",
  "rain_management": ["tip 1", "tip 2"],
  "drought_management": ["tip 1", "tip 2"],
  "crop_management": ["tip 1", "tip 2"],
  "seasonal_tips": ["tip 1"],
  "food_security_alert": "null OR a short alert if food insecurity risk is high",
  "confidence": "high | medium | low",
  "reasoning": "2-3 sentences explaining the key factors driving this advice"
}"""


def _build_user_message(record: dict, rain: dict, drought: dict,
                         health: float, yield_est: float, pest: dict,
                         weather_source: str = "sensor") -> str:
    """Construct the LLM prompt from all model outputs + farm context."""

    rain_pct  = round(rain.get("probability", 0) * 100)
    no_rain   = round((1 - rain.get("probability", 0)) * 100)

    return f"""
=== REAL-TIME FARM REPORT ===
Date/Time  : {record.get("_fetched_at", "now")}
Location   : {record.get("region", "Unknown")} Province — {record.get("_station", "")}
Weather Source: {weather_source}
Crop       : {record.get("crop_type", "unknown")} on {record.get("soil_type", "unknown")} soil
Season     : {record.get("season", "unknown")}

=== LIVE WEATHER MEASUREMENTS ===
Temperature      : {record.get("temperature_c")}°C
Humidity         : {record.get("humidity_pct")}%
Today's Rainfall : {record.get("rainfall_mm")} mm
Soil Moisture    : {record.get("soil_moisture_pct")}%
Wind Speed       : {record.get("wind_speed_kmh")} km/h
Solar Radiation  : {record.get("solar_radiation_wm2")} W/m²
NDVI (vegetation): {record.get("ndvi")} (0=bare, 1=dense)
Forecast rain tomorrow: {record.get("_rain_tomorrow_forecast_mm", "N/A")} mm

=== AI MODEL PREDICTIONS ===
RAIN MODEL (RF + MLP ensemble):
  Rain tomorrow probability : {rain_pct}%  (no rain: {no_rain}%)
  RF confidence: {round(rain.get("rf_probability",0)*100)}% | MLP confidence: {round(rain.get("mlp_probability",0)*100)}%

DROUGHT MODEL (Gradient Boosting + LSTM):
  Drought index  : {drought.get("score")} / 1.0
  Risk level     : {drought.get("level", "unknown").upper()}
  Assessment     : {drought.get("message", "")}

CROP HEALTH MODEL (PyTorch Multi-output DNN):
  Health score   : {round(health, 1)} / 100
  Yield estimate : {round(yield_est)} kg/ha
  Pest pressure  : {pest.get("level", "unknown").upper()} ({pest.get("pest_pressure_index")})
  Pest note      : {pest.get("message", "")}

=== FARMING CONTEXT ===
Fertilizer applied : {record.get("fertilizer_applied_kg_ha")} kg/ha
Irrigation applied : {record.get("irrigation_applied_mm")} mm

Based on all the above, provide your structured JSON advice for this farmer.
"""


# ── Main Groq advisor call ────────────────────────────────────────────────────

def generate_groq_advice(record: dict, rain: dict, drought: dict,
                          health: float, yield_est: float, pest: dict) -> dict:
    """
    Call Groq LLM to generate advice. Returns parsed advice dict.
    Falls back to None on error (caller uses rule-based fallback).
    """
    if not GROQ_API_KEY:
        log.warning("[Groq] API key not set — skipping LLM advice.")
        return None

    try:
        client = _get_client()
        weather_source = record.get("_source", "sensor")
        user_msg = _build_user_message(
            record, rain, drought, health, yield_est, pest, weather_source
        )

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.3,      # low temp = consistent, practical advice
            max_tokens=1024,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        advice = json.loads(raw)

        # Normalize keys so downstream code works regardless of exact LLM output
        advice.setdefault("rain_management",     [])
        advice.setdefault("drought_management",  [])
        advice.setdefault("crop_management",     [])
        advice.setdefault("seasonal_tips",       [])
        advice.setdefault("food_security_alert", None)
        advice.setdefault("confidence",          "medium")
        advice.setdefault("reasoning",           "")

        # Usage metadata
        usage = response.usage
        advice["_groq_meta"] = {
            "model":        GROQ_MODEL,
            "prompt_tokens":response.usage.prompt_tokens,
            "output_tokens":response.usage.completion_tokens,
            "latency_ms":   None,   # Groq doesn't expose this in the SDK currently
        }

        log.info(f"[Groq] Advice generated | "
                 f"tokens: {usage.prompt_tokens}+{usage.completion_tokens}")
        return advice

    except Exception as e:
        log.error(f"[Groq] Error: {e}")
        return None


# ── Enhance existing rule-based advice with Groq ──────────────────────────────

def enhance_advice_with_groq(rule_based_advice: dict, groq_advice: dict | None) -> dict:
    """
    Merge rule-based advice with Groq's LLM advice.
    Groq output takes priority for priority_action, reasoning, food_security_alert.
    All advice lists are combined (Groq first).
    """
    if groq_advice is None:
        rule_based_advice["_advisor"] = "rule-based (Groq unavailable)"
        return rule_based_advice

    merged = dict(rule_based_advice)

    # LLM-generated list advice (prepend so they appear first)
    for key in ["rain_management", "drought_management", "crop_management", "seasonal_tips"]:
        llm_tips  = groq_advice.get(key, [])
        rule_tips = rule_based_advice.get("advice", {}).get(key, [])
        # Deduplicate while preserving order
        seen, combined = set(), []
        for tip in llm_tips + rule_tips:
            if tip not in seen:
                seen.add(tip)
                combined.append(tip)
        merged.setdefault("advice", {})[key] = combined

    # LLM-only fields
    merged["priority_action"]     = groq_advice.get("priority_action",
                                      rule_based_advice.get("priority_action", ""))
    merged["food_security_alert"] = groq_advice.get("food_security_alert")
    merged["groq_reasoning"]      = groq_advice.get("reasoning", "")
    merged["groq_confidence"]     = groq_advice.get("confidence", "medium")
    merged["_advisor"]            = f"Groq ({GROQ_MODEL}) + rule-based"
    merged["_groq_meta"]          = groq_advice.get("_groq_meta", {})

    return merged
