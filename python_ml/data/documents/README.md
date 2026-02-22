# Monthly Forecast Documents — Data Folder

This folder holds **official monthly weather forecast documents** downloaded from
[Rwanda Meteorology Agency (Meteorwanda)](https://www.meteorwanda.gov.rw/products/monthly-forecast).

When you run `python models/train_models.py`, the training pipeline automatically
scans this folder, parses every PDF and DOCX file it finds, and merges the
extracted climate data with the synthetic training dataset.  
Real document data is weighted **2.5×** over synthetic data so models learn
from real-world Rwanda climate signals.

---

## Folder structure

```
data/documents/
└── monthly_forecasts/
    ├── 2020/          ← put files from 2020 here
    │   ├── january_2020.pdf
    │   ├── february_2020.pdf
    │   └── ...
    ├── 2021/
    ├── 2022/
    ├── 2023/
    ├── 2024/
    ├── 2025/
    └── 2026/
```

---

## How to download the documents

1. Go to https://www.meteorwanda.gov.rw/products/monthly-forecast  
2. Select a year (2020 → 2026) and download each monthly bulletin as **PDF** or **DOCX**.  
3. Save each file into the matching year folder, e.g.:
   - `monthly_forecasts/2023/march_2023.pdf`
   - `monthly_forecasts/2024/october_2024.docx`

### Filename convention

Metorwanda publishes files named like:
```
Weather_Forecast_for_December_2025.pdf
Weather_Forecast_for_January_2026.pdf
```
Save them exactly as downloaded — the parser reads the month name and year
directly from the filename.

Other patterns that also work:

| Filename | Detected period |
|---|---|
| `Weather_Forecast_for_December_2025.pdf` | December 2025 ✓ primary format |
| `Weather_Forecast_for_January_2026.pdf` | January 2026 |
| `january_2023.pdf` | January 2023 |
| `2024_april.pdf` | April 2024 |
| `2025-07.docx` | July 2025 |
| `monthly_forecast_feb_2022.pdf` | February 2022 |

If neither the filename nor the document text contains a year/month, the parser
logs a warning and defaults to today's year/month.  **Name your files clearly!**

---

## What the parser extracts

| Column | Source |
|---|---|
| `rainfall_mm` | Tables or text mentioning mm values |
| `temperature_c` | Tables or text mentioning °C values |
| `drought_index` | Derived from "*below/above/normal*" keywords and rainfall ratio |
| `rain_tomorrow` | Derived from rainfall_mm (> 20 mm → 1) |
| `region` | Detected from region names in tables/text (Northern, Southern, Eastern, Western, Kigali) |

Columns not present in the document (soil type, crop type, NDVI, etc.) are
filled automatically from per-region medians computed from the synthetic data.

---

## Installing the required libraries

```bash
pip install pdfplumber python-docx
```

These are already listed in `requirements.txt`.

---

## Testing the parser manually

```bash
# parse a single file
python data/document_parser.py data/documents/monthly_forecasts/2023/march_2023.pdf

# parse the whole folder
python data/document_parser.py data/documents/monthly_forecasts/

# test the integrator (merge + stats)
python data/document_integrator.py
```

---

## Re-training after adding documents

```bash
cd python_ml
python models/train_models.py
```

The pipeline will automatically detect your new files and include them.  
The merged dataset is saved to `data/rwanda_agri_climate_merged.csv` for
inspection.
