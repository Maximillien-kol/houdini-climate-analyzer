// src/services/mlService.js
// Communicates with the Python Flask ML service.

const axios  = require("axios");
const logger = require("../utils/logger");

const ML_BASE = process.env.ML_SERVICE_URL || "http://localhost:5001";
const TIMEOUT = parseInt(process.env.ML_TIMEOUT_MS || "15000");

const client = axios.create({
  baseURL: ML_BASE,
  timeout: TIMEOUT,
  headers: { "Content-Type": "application/json" },
});

// ── Interceptors ──────────────────────────────────────────────────────────────
client.interceptors.request.use((config) => {
  logger.debug(`ML → ${config.method.toUpperCase()} ${config.url}`);
  return config;
});

client.interceptors.response.use(
  (res) => res,
  (err) => {
    const msg = err.response?.data?.error || err.message;
    logger.error(`ML Service error: ${msg}`);
    throw new Error(`ML Service unavailable: ${msg}`);
  }
);

// ── Service calls ──────────────────────────────────────────────────────────────

/**
 * Check if the Python ML service is alive.
 */
async function checkHealth() {
  const { data } = await client.get("/health");
  return data;
}

/**
 * Predict rain for the next day.
 * @param {Object} record - sensor/satellite reading
 */
async function predictRain(record) {
  const { data } = await client.post("/predict/rain", record);
  return data;
}

/**
 * Predict drought risk level and index.
 */
async function predictDrought(record) {
  const { data } = await client.post("/predict/drought", record);
  return data;
}

/**
 * Predict crop health score, yield estimate, and pest risk.
 */
async function predictCrop(record) {
  const { data } = await client.post("/predict/crop", record);
  return data;
}

/**
 * Full pipeline: all models + advice report.
 * Primary endpoint for farmer-facing advice.
 */
async function predictFull(record) {
  const { data } = await client.post("/predict/full", record);
  return data;
}

/**
 * Get full AI report using REAL Open-Meteo weather data (no body needed).
 * @param {string} queryString - e.g. "region=Northern&crop_type=maize"
 */
async function getRealtime(queryString = "") {
  const url = `/predict/realtime${queryString ? "?" + queryString : ""}`;
  const { data } = await client.get(url);
  return data;
}

/**
 * Submit feedback to the ML service for drift monitoring.
 */
async function submitFeedback(feedbackId, actualOutcome) {
  const { data } = await client.post("/feedback", {
    feedback_id: feedbackId,
    actual_outcome: actualOutcome,
  });
  return data;
}

/**
 * Get aggregated feedback / accuracy summary.
 */
async function getFeedbackSummary() {
  const { data } = await client.get("/feedback/summary");
  return data;
}

/**
 * Run a 5-year Holt-Winters climate forecast.
 * @param {string} region    - Rwanda province (Northern/Eastern/Southern/Western/Kigali)
 * @param {number} startYear - first historical year (default 2015)
 */
async function getForecast(region = "Northern", startYear = 2015) {
  const { data } = await client.get(`/forecast/${region}`, {
    params: { start_year: startYear },
    timeout: 60000,          // forecast takes ~10s for Open-Meteo + model training
  });
  return data;
}

module.exports = {
  getForecast,
  checkHealth,
  predictRain,
  predictDrought,
  predictCrop,
  predictFull,
  getRealtime,
  submitFeedback,
  getFeedbackSummary,
};
