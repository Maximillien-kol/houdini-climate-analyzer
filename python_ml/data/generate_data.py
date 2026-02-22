"""
generate_data.py
Simulates data as if collected from sensors, satellites, and drones
covering Rwanda's agricultural regions.
Produces a CSV that the AI pipeline will consume.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── constants ────────────────────────────────────────────────────────────────
RWANDA_REGIONS = [
    "Kigali", "Northern", "Southern", "Eastern", "Western"
]

CROP_TYPES = ["maize", "beans", "sorghum", "cassava", "sweet_potato"]

SOIL_TYPES = ["clay", "loam", "sandy", "volcanic"]

SEASONS = {          # month → season
    1: "dry", 2: "dry",
    3: "rainy_A", 4: "rainy_A", 5: "rainy_A",
    6: "dry", 7: "dry", 8: "dry",
    9: "rainy_B", 10: "rainy_B", 11: "rainy_B",
    12: "dry",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def seasonal_rain(month: int) -> float:
    """Return a monthly rainfall baseline (mm) that mirrors Rwanda's bimodal pattern."""
    rain_map = {
        1: 50, 2: 70, 3: 130, 4: 160, 5: 120,
        6: 30, 7: 15, 8: 20, 9: 80, 10: 110, 11: 120, 12: 60
    }
    return rain_map[month]


def temperature_by_altitude(region: str, month: int) -> float:
    """Simulate temperature (°C) with altitude-aware offsets."""
    altitude_offset = {
        "Kigali": 0, "Northern": -2, "Southern": -1,
        "Eastern": 2, "Western": -1
    }
    base = 18 + 4 * np.sin(2 * np.pi * (month - 3) / 12)
    return round(base + altitude_offset[region] + np.random.normal(0, 1.5), 2)


# ── main generation ──────────────────────────────────────────────────────────

def generate_dataset(n_days: int = 1095, output_path: str = None) -> pd.DataFrame:
    """
    Generate n_days of synthetic agricultural / climate records.

    Columns
    -------
    date, region, crop_type, soil_type,
    temperature_c, humidity_pct, rainfall_mm, soil_moisture_pct,
    wind_speed_kmh, solar_radiation_wm2,
    ndvi (vegetation index), pest_pressure_index,
    fertilizer_applied_kg_ha, irrigation_applied_mm,
    drought_index,          # 0=normal … 1=severe drought
    rain_tomorrow,          # 1 = rain expected next day
    crop_health_score,      # 0–100
    yield_kg_ha             # target for regression
    """

    start_date = datetime(2022, 1, 1)
    records = []

    for day_offset in range(n_days):
        date = start_date + timedelta(days=day_offset)
        month = date.month

        for region in RWANDA_REGIONS:
            crop = np.random.choice(CROP_TYPES)
            soil = np.random.choice(SOIL_TYPES)

            # ── climate variables ─────────────────────────────────────────
            rain_base = seasonal_rain(month)
            rainfall = max(0, np.random.normal(rain_base, rain_base * 0.4))
            temperature = temperature_by_altitude(region, month)
            humidity = np.clip(40 + rainfall * 0.3 + np.random.normal(0, 5), 20, 100)
            wind_speed = abs(np.random.normal(12, 4))
            solar_rad = max(100, np.random.normal(220, 40) - rainfall * 0.5)

            # ── soil & vegetation ─────────────────────────────────────────
            soil_moisture = np.clip(rainfall * 0.6 + np.random.normal(20, 5), 5, 95)
            ndvi = np.clip(0.3 + soil_moisture * 0.004 + np.random.normal(0, 0.05), 0.1, 0.9)

            # ── pest pressure (higher in humid conditions) ────────────────
            pest_pressure = np.clip(
                0.1 + 0.005 * humidity + np.random.normal(0, 0.05), 0, 1
            )

            # ── farming inputs ────────────────────────────────────────────
            fertilizer = np.random.choice([0, 25, 50, 75, 100], p=[0.3, 0.25, 0.2, 0.15, 0.1])
            irrigation = max(0, np.random.normal(0, 5)) if rainfall < 20 else 0

            # ── drought index (SPI proxy) ─────────────────────────────────
            drought_index = np.clip(
                1 - (rainfall / (rain_base + 1)) + np.random.normal(0, 0.1), 0, 1
            )

            # ── next-day rain label ───────────────────────────────────────
            next_day_date = date + timedelta(days=1)
            next_rain_base = seasonal_rain(next_day_date.month)
            rain_tomorrow = int(np.random.binomial(1, min(next_rain_base / 160, 0.95)))

            # ── crop health (0-100) ───────────────────────────────────────
            crop_health = np.clip(
                50
                + ndvi * 30
                - pest_pressure * 25
                + min(soil_moisture, 50) * 0.3
                - drought_index * 20
                + fertilizer * 0.1
                + np.random.normal(0, 3),
                0, 100
            )

            # ── yield (kg/ha) ─────────────────────────────────────────────
            yield_kg_ha = max(0,
                crop_health * 25
                + fertilizer * 3
                - drought_index * 800
                + irrigation * 10
                + np.random.normal(0, 150)
            )

            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "region": region,
                "season": SEASONS[month],
                "crop_type": crop,
                "soil_type": soil,
                "temperature_c": round(temperature, 2),
                "humidity_pct": round(humidity, 2),
                "rainfall_mm": round(rainfall, 2),
                "soil_moisture_pct": round(soil_moisture, 2),
                "wind_speed_kmh": round(wind_speed, 2),
                "solar_radiation_wm2": round(solar_rad, 2),
                "ndvi": round(ndvi, 4),
                "pest_pressure_index": round(pest_pressure, 4),
                "fertilizer_applied_kg_ha": fertilizer,
                "irrigation_applied_mm": round(irrigation, 2),
                "drought_index": round(drought_index, 4),
                "rain_tomorrow": rain_tomorrow,
                "crop_health_score": round(crop_health, 2),
                "yield_kg_ha": round(yield_kg_ha, 2),
            })

    df = pd.DataFrame(records)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[DataGen] Saved {len(df):,} records → {output_path}")

    return df


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "rwanda_agri_climate.csv")
    df = generate_dataset(n_days=1095, output_path=out)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(df.dtypes)
