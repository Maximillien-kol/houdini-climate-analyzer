# AgriShield AI
### Food Insecurity & Agricultural Vulnerability to Climate Shocks

An end-to-end AI system for Rwandan farmers — collects climate and crop
data, runs three machine-learning models, and delivers concrete,
prioritised advice through a REST API.

---

## Quick Start (one command)

```bash
python run.py
```

This single command will:
1. Install all Python and Node.js dependencies
2. Train all AI models — automatically includes any PDF/DOCX forecast
   documents found in `python_ml/data/documents/monthly_forecasts/`
3. Start the Python ML service on **port 5001**
4. Start the Node.js API gateway on **port 3000**

Press **Ctrl+C** to stop both services.

### Other commands

```bash
python run.py retrain    # re-parse documents + retrain models
python run.py serve      # start both servers (models already trained)
python run.py install    # install dependencies only
python run.py train      # synthetic-only training (no documents)
```

### Adding real Meteorwanda forecast documents

Download monthly PDFs from https://www.meteorwanda.gov.rw/products/monthly-forecast
and place them in the matching year folder:

```
python_ml/data/documents/monthly_forecasts/
├── 2025/
│   ├── Weather_Forecast_for_November_2025.pdf
│   └── Weather_Forecast_for_December_2025.pdf
└── 2026/
    ├── Weather_Forecast_for_January_2026.pdf
    └── Weather_Forecast_for_February_2026.pdf
```

Then retrain:

```bash
python run.py retrain
```

Real document rows are weighted **2.5×** over synthetic data so models
learn from official Rwanda climate signals.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                     DATA SOURCES                    │
│   Weather Sensors · Satellites · Drones · Manual    │
└──────────────────────┬──────────────────────────────┘
                       │ JSON sensor readings
                       ▼
┌──────────────────────────────────────────────────────┐
│          NODE.JS API GATEWAY  (port 3000)            │
│  ┌─────────────┐  ┌───────────────┐  ┌───────────┐  │
│  │ /predict/*  │  │  /farmers/*   │  │/feedback/ │  │
│  │  Validate   │  │  Registry     │  │ Tracking  │  │
│  │  Route      │  │  History      │  │ Drift     │  │
│  └──────┬──────┘  └───────────────┘  └───────────┘  │
└─────────┼────────────────────────────────────────────┘
          │ HTTP (axios)
          ▼
┌──────────────────────────────────────────────────────┐
│       PYTHON FLASK ML SERVICE  (port 5001)           │
│                                                      │
│  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │ Rain Prediction  │  │    Drought Risk Model     │  │
│  │ Random Forest    │  │  Gradient Boosting        │  │
│  │ + PyTorch MLP    │  │  + PyTorch LSTM           │  │
│  │ → rain_tomorrow  │  │  → drought_index [0-1]    │  │
│  └──────────────────┘  └──────────────────────────┘  │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │ Crop Health & Yield DNN (PyTorch)            │    │
│  │ Multi-output: health_score + yield_kg_ha     │    │
│  │ + Rule-based Pest Risk Classifier            │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │  Advice Generator  →  Prioritised Action     │    │
│  │  Feedback Logger   →  Drift Monitoring       │    │
│  └──────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

---

## Manual Start (advanced)

If you prefer to run each component individually:

```bash
# 1 — Python dependencies
cd python_ml
pip install -r requirements.txt

# 2 — Train models (includes documents if present)
python main.py retrain

# 3 — Start Python ML service
python main.py serve

# 4 — In a second terminal: Node.js API
cd nodejs_api
npm install
npm start
```

---

## API Reference (Node.js — port 3000)

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Node.js service health |
| GET | `/api/predict/ml-health` | Python ML service health |

### Predictions
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict/full` | **Primary**: full AI report + advice |
| POST | `/api/predict/rain` | Rain probability only |
| POST | `/api/predict/drought` | Drought index + risk level |
| POST | `/api/predict/crop` | Crop health + yield estimate |

### Request body (POST /api/predict/full)
```json
{
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
  "farmer_id": "optional-uuid"
}
```

### Response
```json
{
  "success": true,
  "report": {
    "timestamp": "2026-02-22T10:00:00Z",
    "location": { "region": "Eastern", "crop_type": "maize", "season": "rainy_A" },
    "predictions": {
      "rain":    { "rain_tomorrow": true, "probability": 0.72 },
      "drought": { "level": "mild", "score": 0.30, "message": "..." },
      "crop_health": { "score": 68.5, "yield_estimate_kg_ha": 1400 },
      "pest":    { "level": "low", "pest_pressure_index": 0.18, "message": "..." }
    },
    "advice": {
      "rain_management":    ["..."],
      "drought_management": ["..."],
      "crop_management":    ["..."],
      "seasonal_tips":      ["..."]
    },
    "priority_action": "Rain imminent — postpone field operations.",
    "feedback_id": "A1B2C3D4",
    "summary": { "in_one_line": "...", "rain_status": "...", ... }
  }
}
```

### Farmers
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/farmers` | List all farmers (`?region=&crop_type=`) |
| POST | `/api/farmers` | Register new farmer |
| GET | `/api/farmers/:id` | Get farmer by ID |
| GET | `/api/farmers/:id/history` | Farmer's prediction history |

### Feedback (Track Results / Learn)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/feedback` | Submit real-world outcome |
| GET | `/api/feedback/summary` | Model accuracy / drift report |

### Reference Data
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/data/regions` | Rwanda regions & districts |
| GET | `/api/data/crops` | Supported crop types |
| GET | `/api/data/soils` | Soil types |
| GET | `/api/data/simulate` | Simulate a sensor reading |

---

## How the System Works (Step by Step)

1. **Collect Data** — Sensors/satellites/drones gather weather, soil,
   crop, and pest data. The `/api/data/simulate` endpoint mimics this.

2. **Send to AI** — Data is POSTed to `/api/predict/full` on the
   Node.js gateway, which validates the payload.

3. **Analyse Data** — Three models run in the Python ML service:
   - Random Forest + MLP predicts `rain_tomorrow`
   - Gradient Boosting + LSTM estimates `drought_index`
   - Multi-output DNN scores `crop_health` and `yield_estimate`

4. **Give Advice** — The Advice Generator converts predictions into
   plain-language, prioritised, actionable recommendations.

5. **Track Results** — Farmers report real outcomes (did it rain?
   what was the actual yield?) via `/api/feedback`. The system logs
   these and reports prediction accuracy trends.

---

## Project Structure
```
HOODINI/
├── run.py                            ← master launcher (start here)
│
├── python_ml/
│   ├── data/
│   │   ├── generate_data.py          ← synthetic Rwanda dataset
│   │   ├── document_parser.py        ← PDF/DOCX forecast parser
│   │   ├── document_integrator.py    ← merges real docs + synthetic data
│   │   ├── rwanda_agri_climate.csv   ← generated after retrain
│   │   ├── rwanda_agri_climate_merged.csv  ← merged dataset (real + synthetic)
│   │   └── documents/
│   │       ├── README.md             ← how to add forecast PDFs
│   │       └── monthly_forecasts/
│   │           ├── 2020/ … 2024/     ← drop historical PDFs here
│   │           ├── 2025/             ← Weather_Forecast_for_*.pdf
│   │           └── 2026/
│   ├── models/
│   │   ├── rain_prediction_model.py  ← RF + MLP
│   │   ├── drought_risk_model.py     ← GB + LSTM
│   │   ├── crop_health_model.py      ← multi-output DNN
│   │   └── train_models.py           ← master training script
│   ├── utils/
│   │   ├── data_processor.py         ← encode, scale, split
│   │   └── advice_generator.py       ← rules + feedback log
│   ├── api/
│   │   └── prediction_service.py     ← Flask REST API
│   ├── artifacts/                    ← saved models (created at retrain)
│   ├── requirements.txt
│   └── main.py                       ← CLI entry point
│
└── nodejs_api/
    ├── src/
    │   ├── app.js                    ← Express configuration
    │   ├── routes/
    │   │   ├── predictions.js
    │   │   ├── farmers.js
    │   │   ├── data.js
    │   │   └── feedback.js
    │   ├── services/
    │   │   ├── mlService.js          ← HTTP client → Python
    │   │   └── dataService.js        ← farmer registry + history
    │   └── utils/
    │       └── logger.js
    ├── server.js
    ├── package.json
    └── .env.example
```

---

## Models & Algorithms

| Task | Algorithm | Library |
|------|-----------|---------|
| Rain Prediction | Random Forest + Isotonic Calibration | scikit-learn |
| Rain Prediction (ensemble) | MLP (3 layers, BatchNorm, Dropout) | PyTorch |
| Drought Risk | Gradient Boosting Regressor | scikit-learn |
| Drought Tracking | LSTM (window=7 days) | PyTorch |
| Crop Health + Yield | Multi-output DNN (shared backbone) | PyTorch |
| Pest Risk | Rule-based threshold classifier | Python |
| Advice | Rule-based + model-output aggregator | Python |

---

## Rwanda Context
- **Regions**: Kigali, Northern, Southern, Eastern, Western
- **Key crops**: Maize, Beans, Sorghum, Cassava, Sweet Potato
- **Seasons**: Season A (Mar–May), Season B (Sep–Nov), Dry (Jun–Aug, Dec–Feb)
- **Reference**: Rwanda Agriculture Board (RAB) crop calendars
