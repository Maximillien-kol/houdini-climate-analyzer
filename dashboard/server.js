/**
 * AgriShield Dashboard Server
 * Proxies requests to the HuggingFace Space API to avoid CORS issues,
 * then serves the static dashboard UI.
 */

const express = require("express");
const axios   = require("axios");
const path    = require("path");

const app  = express();
const PORT = process.env.PORT || 4000;

// ── Point this at your HF Space (or localhost:5001 for local dev) ─────────────
const ML_API = process.env.ML_API_URL || "https://max1m1ll1en-rwac-v-0-1.hf.space";

app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// ── Proxy: GET /api/health ────────────────────────────────────────────────────
app.get("/api/health", async (req, res) => {
  try {
    const r = await axios.get(`${ML_API}/health`, { timeout: 15000 });
    res.json(r.data);
  } catch (e) {
    res.status(503).json({ status: "unreachable", error: e.message });
  }
});

// ── Proxy: GET /api/realtime/:region ─────────────────────────────────────────
app.get("/api/realtime/:region", async (req, res) => {
  try {
    const r = await axios.get(`${ML_API}/predict/realtime/${req.params.region}`, { timeout: 30000 });
    res.json(r.data);
  } catch (e) {
    res.status(502).json({ error: e.message });
  }
});

// ── Proxy: POST /api/predict ──────────────────────────────────────────────────
app.post("/api/predict", async (req, res) => {
  try {
    const r = await axios.post(`${ML_API}/predict/full`, req.body, { timeout: 30000 });
    res.json(r.data);
  } catch (e) {
    const status = e.response?.status || 502;
    res.status(status).json({ error: e.response?.data || e.message });
  }
});

// ── Proxy: GET /api/forecast/:region ─────────────────────────────────────────
app.get("/api/forecast/:region", async (req, res) => {
  try {
    const r = await axios.get(`${ML_API}/forecast/${req.params.region}`, { timeout: 30000 });
    res.json(r.data);
  } catch (e) {
    res.status(502).json({ error: e.message });
  }
});

// ── Proxy: GET /api/monthly/:region — 12-month ML forecast ───────────────────
app.get("/api/monthly/:region", async (req, res) => {
  try {
    const { region }   = req.params;
    const crop_type    = req.query.crop_type    || "maize";
    const fertilizer   = req.query.fertilizer   || "50";
    const irrigation   = req.query.irrigation   || "0";
    const url = `${ML_API}/predict/monthly/${region}?crop_type=${crop_type}&fertilizer_kg_ha=${fertilizer}&irrigation_mm=${irrigation}`;
    const r   = await axios.get(url, { timeout: 45000 });
    res.json(r.data);
  } catch (e) {
    const status = e.response?.status || 502;
    res.status(status).json({ error: e.response?.data || e.message });
  }
});

// ── Fallback: serve dashboard for any unknown GET ─────────────────────────────
app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.listen(PORT, () => {
  console.log(`\n  AgriShield Dashboard running at http://localhost:${PORT}`);
  console.log(`  Proxying ML API → ${ML_API}\n`);
});
