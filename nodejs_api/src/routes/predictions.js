// src/routes/predictions.js
// Routes for all AI prediction endpoints.
// Node.js validates input, forwards to Python ML service, enriches response.

const express   = require("express");
const { body, validationResult } = require("express-validator");

const mlService   = require("../services/mlService");
const dataService = require("../services/dataService");
const logger      = require("../utils/logger");

const router = express.Router();

// ── Validation schema ─────────────────────────────────────────────────────────
const climateFields = [
  body("region").isString().notEmpty().withMessage("region is required"),
  body("season").isIn(["rainy_A", "rainy_B", "dry"]).withMessage("season must be rainy_A, rainy_B, or dry"),
  body("crop_type").isString().notEmpty(),
  body("soil_type").isString().notEmpty(),
  body("temperature_c").isFloat({ min: -10, max: 50 }),
  body("humidity_pct").isFloat({ min: 0, max: 100 }),
  body("rainfall_mm").isFloat({ min: 0 }),
  body("soil_moisture_pct").isFloat({ min: 0, max: 100 }),
  body("wind_speed_kmh").isFloat({ min: 0 }),
  body("solar_radiation_wm2").isFloat({ min: 0 }),
  body("ndvi").isFloat({ min: 0, max: 1 }),
  body("pest_pressure_index").isFloat({ min: 0, max: 1 }),
  body("fertilizer_applied_kg_ha").isFloat({ min: 0 }),
  body("irrigation_applied_mm").isFloat({ min: 0 }),
  body("drought_index").isFloat({ min: 0, max: 1 }),
];

function validate(req, res, next) {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(422).json({
      success: false,
      errors: errors.array().map((e) => ({ field: e.path, message: e.msg })),
    });
  }
  next();
}

// ── GET /api/forecast & /api/forecast/:region ───────────────────────────────
router.get("/forecast", async (req, res) => {
  try {
    const region    = req.query.region    || "Northern";
    const startYear = req.query.start_year || 2015;
    logger.info(`Forecast request: region=${region}, start_year=${startYear}`);
    const result = await mlService.getForecast(region, startYear);
    res.json(result);
  } catch (err) {
    logger.error(`/forecast: ${err.message}`);
    res.status(503).json({ success: false, error: err.message });
  }
});

router.get("/forecast/:region", async (req, res) => {
  try {
    const region    = req.params.region;
    const startYear = req.query.start_year || 2015;
    logger.info(`Forecast request: region=${region}, start_year=${startYear}`);
    const result = await mlService.getForecast(region, startYear);
    res.json(result);
  } catch (err) {
    logger.error(`/forecast/${req.params.region}: ${err.message}`);
    res.status(503).json({ success: false, error: err.message });
  }
});

// ── POST /api/predict/rain ────────────────────────────────────────────────────
router.post("/rain", climateFields, validate, async (req, res) => {
  try {
    const result = await mlService.predictRain(req.body);
    res.json(result);
  } catch (err) {
    logger.error(`/predict/rain: ${err.message}`);
    res.status(502).json({ success: false, error: err.message });
  }
});

// ── POST /api/predict/drought ─────────────────────────────────────────────────
router.post("/drought", climateFields, validate, async (req, res) => {
  try {
    const result = await mlService.predictDrought(req.body);
    res.json(result);
  } catch (err) {
    logger.error(`/predict/drought: ${err.message}`);
    res.status(502).json({ success: false, error: err.message });
  }
});

// ── POST /api/predict/crop ────────────────────────────────────────────────────
router.post("/crop", climateFields, validate, async (req, res) => {
  try {
    const result = await mlService.predictCrop(req.body);
    res.json(result);
  } catch (err) {
    logger.error(`/predict/crop: ${err.message}`);
    res.status(502).json({ success: false, error: err.message });
  }
});

// ── POST /api/predict/full ────────────────────────────────────────────────────
// Primary endpoint: full AI report + advice for a farmer.
// Optional: pass farmer_id to save prediction to history.
router.post(
  "/full",
  [
    ...climateFields,
    body("farmer_id").optional().isString(),
  ],
  validate,
  async (req, res) => {
    try {
      const { farmer_id, ...record } = req.body;

      // Get full AI report from Python service
      const result = await mlService.predictFull(record);

      // Optionally save to farmer history
      if (farmer_id) {
        const farmer = dataService.getFarmerById(farmer_id);
        if (farmer) {
          dataService.savePrediction(farmer_id, {
            input: record,
            output: result.report,
          });
        }
      }

      // Enrich response with human-readable summary
      if (result.success && result.report) {
        result.report.summary = buildSummary(result.report);
      }

      res.json(result);
    } catch (err) {
      logger.error(`/predict/full: ${err.message}`);
      res.status(502).json({ success: false, error: err.message });
    }
  }
);

// ── GET /api/predict/realtime  (real Open-Meteo weather + all models + Groq) ──
router.get("/realtime", async (req, res) => {
  try {
    const {
      region = "Northern", crop_type = "maize",
      fertilizer_kg_ha = 50, irrigation_mm = 0, farmer_id,
    } = req.query;

    const qs = new URLSearchParams({ region, crop_type, fertilizer_kg_ha, irrigation_mm }).toString();
    const result = await mlService.getRealtime(qs);

    if (farmer_id && result.success && result.report) {
      const farmer = dataService.getFarmerById(farmer_id);
      if (farmer) {
        dataService.savePrediction(farmer_id, {
          type: "realtime", input: { region, crop_type }, output: result.report,
        });
      }
    }
    if (result.success && result.report) {
      result.report.summary = buildSummary(result.report);
    }
    res.json(result);
  } catch (err) {
    logger.error(`/predict/realtime: ${err.message}`);
    res.status(502).json({ success: false, error: err.message });
  }
});

// ── GET /api/predict/realtime/:region  (shorthand per province) ───────────────
router.get("/realtime/:region", async (req, res) => {
  try {
    const { region } = req.params;
    const { crop_type = "maize", fertilizer_kg_ha = 50, irrigation_mm = 0 } = req.query;
    const qs = new URLSearchParams({ region, crop_type, fertilizer_kg_ha, irrigation_mm }).toString();
    const result = await mlService.getRealtime(qs);
    if (result.success && result.report) {
      result.report.summary = buildSummary(result.report);
    }
    res.json(result);
  } catch (err) {
    logger.error(`/predict/realtime/${req.params.region}: ${err.message}`);
    res.status(502).json({ success: false, error: err.message });
  }
});

// ── ML service health (proxy) ──────────────────────────────────────────────────
router.get("/ml-health", async (req, res) => {
  try {
    const result = await mlService.checkHealth();
    res.json(result);
  } catch (err) {
    res.status(502).json({ success: false, error: "ML service is unreachable." });
  }
});

// ── Helpers ───────────────────────────────────────────────────────────────────
function buildSummary(report) {
  const rain     = report.predictions?.rain;
  const drought  = report.predictions?.drought;
  const health   = report.predictions?.crop_health;
  const priority = report.priority_action;

  return {
    in_one_line: priority,
    rain_status: rain?.rain_tomorrow
      ? `Rain expected (${Math.round(rain.probability * 100)}%)`
      : `No rain (${Math.round((1 - rain?.probability) * 100)}% dry)`,
    drought_status: `${drought?.level || "unknown"} drought risk (score: ${drought?.score})`,
    crop_health: `${health?.score}/100 | Est. yield: ${health?.yield_estimate_kg_ha} kg/ha`,
  };
}

module.exports = router;
