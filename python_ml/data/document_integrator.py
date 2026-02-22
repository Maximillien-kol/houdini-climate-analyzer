"""
document_integrator.py
======================
Merges parsed Meteorwanda monthly forecast documents with the synthetic
Rwanda CSV dataset so that real-world climate signals are woven into the
training data.

How it works
------------
1. Reads the synthetic CSV (rwanda_agri_climate.csv).
2. Scans data/documents/ for PDF / DOCX files, parses each one with
   document_parser.py, and converts monthly forecasts into *daily* rows
   (small Gaussian jitter around the monthly mean so distributions stay
   smooth).
3. Fills missing columns (soil type, crop type, NDVI, etc.) using the
   per-region medians from the synthetic data, ensuring the merged DataFrame
   has exactly the same columns as the synthetic one.
4. Marks every document-sourced row with  source="document"  and
   data_weight=<float>  so training code can optionally up-weight real data.
5. Returns a single merged DataFrame ready to pass directly to the model
   trainers.

Usage (standalone)
------------------
    python document_integrator.py          # prints stats, saves merged CSV

Usage (from train_models.py)
-----------------------------
    from data.document_integrator import load_merged_dataset
    df = load_merged_dataset()
"""

from __future__ import annotations

import os
import logging
import calendar
from datetime import date, timedelta

import numpy as np
import pandas as pd

from data.document_parser import parse_documents_folder

logger = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE          = os.path.dirname(__file__)
DOCUMENTS_DIR  = os.path.join(_HERE, "documents")
SYNTHETIC_CSV  = os.path.join(_HERE, "rwanda_agri_climate.csv")
MERGED_CSV     = os.path.join(_HERE, "rwanda_agri_climate_merged.csv")

# Weight given to document-sourced rows during training (relative to 1.0 for synthetic)
DOCUMENT_ROW_WEIGHT = 2.5

# Jitter applied when expanding monthly forecast → daily rows (std dev fraction)
DAILY_JITTER_FRACTION = 0.08


# ═══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def load_merged_dataset(
    synthetic_csv: str = SYNTHETIC_CSV,
    documents_dir: str = DOCUMENTS_DIR,
    save_merged: bool = True,
    merged_csv: str = MERGED_CSV,
) -> pd.DataFrame:
    """
    Load the synthetic CSV and, if any documents exist, parse + merge them.

    Parameters
    ----------
    synthetic_csv   Path to the base synthetic dataset.
    documents_dir   Root folder that is recursively scanned for PDF/DOCX.
    save_merged     If True, write the merged DataFrame to *merged_csv*.
    merged_csv      Output path for the merged CSV.

    Returns
    -------
    pd.DataFrame with all original columns plus:
        source        "synthetic" | "document"
        data_weight   float (1.0 for synthetic, DOCUMENT_ROW_WEIGHT for docs)
    """
    # ── 1. load synthetic base ─────────────────────────────────────────────
    if not os.path.exists(synthetic_csv):
        raise FileNotFoundError(
            f"Synthetic CSV not found: {synthetic_csv}\n"
            "Run python models/train_models.py first to generate it."
        )

    synthetic = pd.read_csv(synthetic_csv, parse_dates=["date"])
    synthetic["source"]      = "synthetic"
    synthetic["data_weight"] = 1.0
    logger.info(f"[Integrator] Synthetic dataset: {len(synthetic):,} rows")

    # ── 2. scan documents folder ───────────────────────────────────────────
    doc_root = os.path.abspath(documents_dir)
    if not os.path.isdir(doc_root):
        logger.warning(
            f"[Integrator] Documents folder not found: {doc_root}\n"
            "  → Training on synthetic data only."
        )
        _ensure_weight_col(synthetic)
        return synthetic

    doc_df = parse_documents_folder(doc_root)

    if doc_df.empty:
        logger.info(
            "[Integrator] No parseable documents found in documents/.\n"
            "  → Training on synthetic data only."
        )
        _ensure_weight_col(synthetic)
        return synthetic

    logger.info(
        f"[Integrator] Parsed {len(doc_df)} monthly (region, month) records "
        f"from documents."
    )

    # ── 3. expand monthly rows → daily rows ────────────────────────────────
    daily_doc = _monthly_to_daily(doc_df, synthetic)
    logger.info(
        f"[Integrator] Expanded to {len(daily_doc):,} daily document rows."
    )

    # ── 4. align columns to synthetic schema ──────────────────────────────
    daily_doc = _align_columns(daily_doc, synthetic)

    # ── 5. mark weights ───────────────────────────────────────────────────
    daily_doc["source"]      = "document"
    daily_doc["data_weight"] = DOCUMENT_ROW_WEIGHT

    # ── 6. merge and remove duplicate (date, region) preferring documents ──
    merged = pd.concat([synthetic, daily_doc], ignore_index=True)
    merged = merged.sort_values(["date", "region", "source"])

    # Where both synthetic and document exist for same date+region, keep doc
    merged = (
        merged
        .sort_values("source", ascending=False)   # "synthetic" < "document" alphabetically
        .drop_duplicates(subset=["date", "region"], keep="first")
        .sort_values(["date", "region"])
        .reset_index(drop=True)
    )

    n_synthetic = (merged["source"] == "synthetic").sum()
    n_document  = (merged["source"] == "document").sum()
    logger.info(
        f"[Integrator] Merged dataset: {len(merged):,} rows  "
        f"({n_synthetic:,} synthetic + {n_document:,} from documents)"
    )

    # ── 7. optionally persist ──────────────────────────────────────────────
    if save_merged:
        merged.to_csv(merged_csv, index=False)
        logger.info(f"[Integrator] Saved merged CSV → {merged_csv}")

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _monthly_to_daily(
    doc_df: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> pd.DataFrame:
    """
    Expand each (year-month, region) forecast row into one row per day of
    that month, applying small Gaussian jitter so distributions stay smooth.
    """
    rng = np.random.default_rng(seed=777)
    day_rows: list[dict] = []

    for _, row in doc_df.iterrows():
        d: date = row["date"].date() if hasattr(row["date"], "date") else row["date"]
        year, month = d.year, d.month
        region      = row["region"]
        n_days      = calendar.monthrange(year, month)[1]

        # Base values (may be NaN)
        rain_base  = row.get("rainfall_mm",   np.nan)
        temp_base  = row.get("temperature_c", np.nan)
        drought_b  = row.get("drought_index", 0.3)
        rt         = row.get("rain_tomorrow", 1 if not np.isnan(rain_base) and rain_base > 20 else 0)

        # Regional medians from synthetic (used to fill NaNs)
        syn_region = synthetic[synthetic["region"] == region]
        syn_month  = syn_region[pd.to_datetime(syn_region["date"]).dt.month == month]
        if syn_month.empty:
            syn_month = syn_region  # fall back to whole-region stats

        rain_ref  = syn_month["rainfall_mm"].median()   if not syn_month.empty else 80.0
        temp_ref  = syn_month["temperature_c"].median() if not syn_month.empty else 19.0

        if np.isnan(rain_base):
            rain_base = rain_ref
        if np.isnan(temp_base):
            temp_base = temp_ref

        for day in range(1, n_days + 1):
            day_date = date(year, month, day)

            jitter_r = rng.normal(0, max(rain_base * DAILY_JITTER_FRACTION, 1.0))
            jitter_t = rng.normal(0, 0.8)

            day_rain = float(np.clip(rain_base / n_days + jitter_r, 0, None))
            day_temp = float(temp_base + jitter_t)

            # derive dependent fields
            humidity = float(np.clip(40 + day_rain * 0.3 + rng.normal(0, 3), 20, 100))
            soil_moi = float(np.clip(day_rain * 0.6 + rng.normal(20, 4), 5, 95))
            ndvi     = float(np.clip(0.3 + soil_moi * 0.004 + rng.normal(0, 0.03), 0.1, 0.9))
            wind     = float(abs(rng.normal(12, 3)))
            solar    = float(np.clip(220 - day_rain * 0.5 + rng.normal(0, 20), 80, 350))
            pest     = float(np.clip(0.1 + 0.005 * humidity + rng.normal(0, 0.03), 0, 1))

            day_drought = float(np.clip(
                drought_b + rng.normal(0, 0.05), 0, 1
            ))

            day_rows.append({
                "date":                   day_date.strftime("%Y-%m-%d"),
                "region":                 region,
                "season":                 row.get("season", "dry"),
                # Document-derived fields
                "rainfall_mm":            round(day_rain, 2),
                "temperature_c":          round(day_temp, 2),
                "humidity_pct":           round(humidity, 2),
                "soil_moisture_pct":      round(soil_moi, 2),
                "wind_speed_kmh":         round(wind, 2),
                "solar_radiation_wm2":    round(solar, 2),
                "ndvi":                   round(ndvi, 4),
                "pest_pressure_index":    round(pest, 4),
                "drought_index":          round(day_drought, 4),
                "rain_tomorrow":          int(rt),
                # Farming inputs – sampled from synthetic medians later
                "fertilizer_applied_kg_ha": np.nan,
                "irrigation_applied_mm":    np.nan,
                "crop_health_score":        np.nan,
                "yield_kg_ha":              np.nan,
                # Categoricals – filled later
                "crop_type": np.nan,
                "soil_type":  np.nan,
            })

    return pd.DataFrame(day_rows)


def _align_columns(doc_daily: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing columns in *doc_daily* using per-region medians / modes
    computed from *synthetic*.  After this function, doc_daily will have
    every column that synthetic has.
    """
    # Compute per-region stats once
    region_stats: dict[str, dict] = {}
    for region, grp in synthetic.groupby("region"):
        region_stats[region] = {
            "fertilizer_applied_kg_ha": grp["fertilizer_applied_kg_ha"].median(),
            "irrigation_applied_mm":    grp["irrigation_applied_mm"].median(),
            "crop_health_score":        grp["crop_health_score"].median(),
            "yield_kg_ha":              grp["yield_kg_ha"].median(),
            "crop_type":                grp["crop_type"].mode().iloc[0] if not grp["crop_type"].mode().empty else "maize",
            "soil_type":                grp["soil_type"].mode().iloc[0] if not grp["soil_type"].mode().empty else "loam",
        }

    def fill_row(row):
        region  = row.get("region", "Kigali")
        stats   = region_stats.get(region, region_stats.get("Kigali", {}))
        for col, default in stats.items():
            if col not in row or (isinstance(row[col], float) and np.isnan(row[col])):
                row[col] = default
        return row

    doc_daily = doc_daily.apply(fill_row, axis=1)

    # Add any synthetic columns still missing
    for col in synthetic.columns:
        if col not in doc_daily.columns:
            s = synthetic[col]
            # Use mode for non-numeric columns (covers both object and StringDtype)
            try:
                doc_daily[col] = s.median()
            except (TypeError, AttributeError):
                doc_daily[col] = s.mode().iloc[0] if not s.mode().empty else None

    return doc_daily[synthetic.columns.tolist()]


def _ensure_weight_col(df: pd.DataFrame) -> None:
    if "data_weight" not in df.columns:
        df["data_weight"] = 1.0
    if "source" not in df.columns:
        df["source"] = "synthetic"


# ═══════════════════════════════════════════════════════════════════════════════
# CLI convenience
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    merged = load_merged_dataset()
    print("\nColumn dtypes:")
    print(merged.dtypes)
    print(f"\nTotal rows : {len(merged):,}")
    print(f"Synthetic  : {(merged['source'] == 'synthetic').sum():,}")
    print(f"Documents  : {(merged['source'] == 'document').sum():,}")
    print("\nSample document rows:")
    doc_rows = merged[merged["source"] == "document"]
    if not doc_rows.empty:
        print(doc_rows.head(10).to_string())
    else:
        print("  (none – add PDFs/DOCXs to data/documents/monthly_forecasts/)")
