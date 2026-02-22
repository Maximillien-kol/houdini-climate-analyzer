"""
weather_fetcher.py
Fetches REAL climate data from the Open-Meteo API.
No API key required — free satellite/NWP weather data for any coordinate.

Covers all Rwanda provinces. Default = Northern Rwanda (Musanze/Gicumbi).

Open-Meteo docs: https://open-meteo.com/en/docs
"""

import os
import json
import datetime
import urllib.request
import urllib.parse
from functools import lru_cache

# ── Province coordinates (centroid of key agricultural districts) ─────────────
PROVINCE_COORDS = {
    "Northern": {"lat": -1.5014, "lon": 29.6344, "name": "Musanze"},
    "Eastern":  {"lat": -1.7835, "lon": 30.4420, "name": "Nyagatare"},
    "Southern": {"lat": -2.5967, "lon": 29.7394, "name": "Huye"},
    "Western":  {"lat": -2.4757, "lon": 28.9070, "name": "Rusizi"},
    "Kigali":   {"lat": -1.9441, "lon": 30.0619, "name": "Kigali"},
}

# ── Season logic ──────────────────────────────────────────────────────────────
SEASON_MAP = {
    1:"dry", 2:"dry", 3:"rainy_A", 4:"rainy_A", 5:"rainy_A",
    6:"dry", 7:"dry", 8:"dry", 9:"rainy_B", 10:"rainy_B", 11:"rainy_B", 12:"dry"
}

SOIL_DEFAULTS = {
    "Northern": "volcanic",
    "Eastern":  "sandy",
    "Southern": "loam",
    "Western":  "volcanic",
    "Kigali":   "clay",
}

FORECAST_API  = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_API = "https://archive-api.open-meteo.com/v1/archive"


def _api_get(base_url: str, params: dict) -> dict:
    url = base_url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "AgriShield-AI/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


# ── NDVI estimation (no free satellite NDVI API without auth) ─────────────────
def _estimate_ndvi(rainfall_mm: float, season: str, temp_c: float) -> float:
    """
    Estimate NDVI from rainfall + season + temperature.
    NDVI ranges 0.1 (bare) → 0.9 (dense vegetation).
    """
    base = 0.35 if season == "dry" else 0.55
    rain_boost = min(rainfall_mm / 200, 0.25)
    temp_factor = max(0, min((temp_c - 12) / 15, 0.1))
    return round(min(0.9, max(0.1, base + rain_boost + temp_factor)), 4)


# ── Pest pressure estimation from humidity + temperature ──────────────────────
def _estimate_pest_pressure(humidity: float, temp: float, season: str) -> float:
    """Empirical pest pressure: higher in warm, humid rainy seasons."""
    base = 0.15 if season == "dry" else 0.25
    hum_factor  = (max(0, humidity - 50)) / 500
    temp_factor = (max(0, temp - 18)) / 100
    return round(min(1.0, max(0.0, base + hum_factor + temp_factor)), 4)


# ── Drought index (simplified SPI from precipitation anomaly) ─────────────────
MONTHLY_NORMALS_MM = {
    1:50, 2:70, 3:130, 4:160, 5:120, 6:30, 7:15, 8:20, 9:80, 10:110, 11:120, 12:60
}

def _drought_index(rainfall_mm: float, month: int, days: int = 2) -> float:
    """
    Compare rainfall over `days` against what's climatologically expected
    for that same number of days. This avoids the bug where a 2-day total
    (e.g. 6 mm) is compared against a full monthly normal (e.g. 70 mm),
    which always produces near-1.0 (extreme) drought.
    """
    monthly_normal = MONTHLY_NORMALS_MM.get(month, 80)
    period_normal  = monthly_normal / 30.0 * days   # expected mm over `days` days
    return round(max(0.0, min(1.0, 1.0 - (rainfall_mm / (period_normal + 1)))), 4)


# ── CURRENT / FORECAST weather ────────────────────────────────────────────────

def fetch_current_weather(region: str = "Northern", crop_type: str = "maize",
                           fertilizer_kg_ha: float = 50,
                           irrigation_mm: float = 0.0) -> dict:
    """
    Fetch today's real weather for the given Rwanda region.
    Returns a dict ready to pass directly to the ML prediction pipeline.
    """
    coords = PROVINCE_COORDS.get(region, PROVINCE_COORDS["Northern"])
    now    = datetime.datetime.utcnow()
    month  = now.month
    season = SEASON_MAP[month]

    params = {
        "latitude":  coords["lat"],
        "longitude": coords["lon"],
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "shortwave_radiation",
            "soil_moisture_0_to_1cm",
        ]),
        "daily": "precipitation_sum",
        "forecast_days": 2,
        "timezone": "Africa/Kigali",
    }

    data = _api_get(FORECAST_API, params)
    cur  = data["current"]

    temp_c      = round(cur.get("temperature_2m", 20.0), 2)
    humidity    = round(min(100, max(0, cur.get("relative_humidity_2m", 60.0))), 2)
    rainfall_mm = round(max(0, cur.get("precipitation", 0.0)), 2)
    wind_kmh    = round(cur.get("wind_speed_10m", 10.0), 2)
    solar_rad   = round(max(0, cur.get("shortwave_radiation", 200.0)), 2)

    # soil moisture from API (m³/m³) → convert to %
    sm_raw = cur.get("soil_moisture_0_to_1cm", 0.25)
    soil_moisture = round(min(100, max(0, sm_raw * 200)), 2)  # rough % conversion

    # tomorrow's precipitation sum for rain_tomorrow label target
    daily = data.get("daily", {})
    precip_sums = daily.get("precipitation_sum", [0, 0])
    rain_tomorrow_mm = precip_sums[1] if len(precip_sums) > 1 else 0

    ndvi          = _estimate_ndvi(rainfall_mm, season, temp_c)
    pest_pressure = _estimate_pest_pressure(humidity, temp_c, season)
    # Use the 2-day forecast total vs 2-day climatological normal.
    # This prevents near-1.0 drought scores simply because it's not
    # raining at this exact moment.
    forecast_total_mm = sum(precip_sums[:2]) if precip_sums else rainfall_mm
    drought_idx   = _drought_index(forecast_total_mm, month, days=2)

    record = {
        # identifiers
        "region":    region,
        "season":    season,
        "crop_type": crop_type,
        "soil_type": SOIL_DEFAULTS.get(region, "loam"),

        # real weather measurements
        "temperature_c":          temp_c,
        "humidity_pct":           humidity,
        "rainfall_mm":            rainfall_mm,
        "soil_moisture_pct":      soil_moisture,
        "wind_speed_kmh":         wind_kmh,
        "solar_radiation_wm2":    solar_rad,

        # derived / estimated
        "ndvi":                   ndvi,
        "pest_pressure_index":    pest_pressure,
        "drought_index":          drought_idx,

        # farming inputs
        "fertilizer_applied_kg_ha": fertilizer_kg_ha,
        "irrigation_applied_mm":    irrigation_mm,

        # metadata
        "_source": "Open-Meteo real-time",
        "_station": coords["name"],
        "_latitude":  coords["lat"],
        "_longitude": coords["lon"],
        "_fetched_at": now.isoformat() + "Z",
        "_rain_tomorrow_forecast_mm": round(rain_tomorrow_mm, 2),
    }

    return record


# ── HISTORICAL data for retraining ────────────────────────────────────────────

def fetch_historical_weather(region: str = "Northern",
                              start_date: str = "2023-01-01",
                              end_date: str = None,
                              crop_type: str = "maize") -> list[dict]:
    """
    Download historical daily weather for a region from Open-Meteo Archive API.
    Returns a list of dicts, one per day, in the same format as fetch_current_weather.
    """
    import time

    coords   = PROVINCE_COORDS.get(region, PROVINCE_COORDS["Northern"])
    end_date = end_date or datetime.date.today().strftime("%Y-%m-%d")

    params = {
        "latitude":   coords["lat"],
        "longitude":  coords["lon"],
        "start_date": start_date,
        "end_date":   end_date,
        "daily": ",".join([
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max",
            "shortwave_radiation_sum",
            "soil_moisture_0_to_7cm_mean",
            "et0_fao_evapotranspiration",
        ]),
        "timezone": "Africa/Kigali",
    }

    print(f"[WeatherFetcher] Fetching historical data {start_date}→{end_date} "
          f"for {region} ({coords['name']}) …")

    data  = _api_get(HISTORICAL_API, params)
    daily = data["daily"]

    dates      = daily.get("time", [])
    temps      = daily.get("temperature_2m_mean", [None]*len(dates))
    humidity   = daily.get("relative_humidity_2m_mean", [None]*len(dates))
    precip     = daily.get("precipitation_sum", [None]*len(dates))
    wind       = daily.get("wind_speed_10m_max", [None]*len(dates))
    solar      = daily.get("shortwave_radiation_sum", [None]*len(dates))
    soil_moist = daily.get("soil_moisture_0_to_7cm_mean", [None]*len(dates))

    records = []
    for i, date_str in enumerate(dates):
        month = int(date_str[5:7])
        season = SEASON_MAP[month]

        t   = temps[i]      or 20.0
        h   = humidity[i]   or 60.0
        r   = precip[i]     or 0.0
        w   = wind[i]       or 10.0
        s   = solar[i]      or 200.0
        sm  = soil_moist[i] or 0.25

        soil_pct = round(min(100, max(0, sm * 200)), 2)

        record = {
            "date":    date_str,
            "region":  region,
            "season":  season,
            "crop_type": crop_type,
            "soil_type": SOIL_DEFAULTS.get(region, "loam"),

            "temperature_c":           round(t, 2),
            "humidity_pct":            round(min(100, max(0, h)), 2),
            "rainfall_mm":             round(max(0, r), 2),
            "soil_moisture_pct":       soil_pct,
            "wind_speed_kmh":          round(max(0, w), 2),
            "solar_radiation_wm2":     round(max(0, s), 2),

            "ndvi":                    _estimate_ndvi(r, season, t),
            "pest_pressure_index":     _estimate_pest_pressure(h, t, season),
            "drought_index":           _drought_index(r, month),

            "fertilizer_applied_kg_ha": 50.0,
            "irrigation_applied_mm":    0.0,

            "_source": "Open-Meteo archive",
            "_station": coords["name"],
        }
        records.append(record)

    print(f"[WeatherFetcher] Retrieved {len(records)} daily records.")
    return records


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── Current Weather for Northern Rwanda (Musanze) ──────────")
    rec = fetch_current_weather("Northern")
    for k, v in rec.items():
        print(f"  {k:35s}: {v}")
