// src/routes/farmers.js
// Farmer registry — register, list, view history.

const express = require("express");
const { body, param, query, validationResult } = require("express-validator");
const dataService = require("../services/dataService");
const logger      = require("../utils/logger");

const router = express.Router();

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

// ── GET /api/farmers ──────────────────────────────────────────────────────────
router.get("/", async (req, res) => {
  try {
    const { region, crop_type } = req.query;
    let farmers = dataService.getAllFarmers();

    if (region)    farmers = farmers.filter((f) => f.region === region);
    if (crop_type) farmers = farmers.filter((f) => f.crop_type === crop_type);

    res.json({
      success: true,
      count: farmers.length,
      farmers,
      stats: dataService.getSystemStats(),
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// ── POST /api/farmers ─────────────────────────────────────────────────────────
router.post(
  "/",
  [
    body("name").isString().notEmpty(),
    body("region").isString().notEmpty(),
    body("district").isString().notEmpty(),
    body("crop_type").isString().notEmpty(),
    body("soil_type").isString().notEmpty(),
    body("farm_size_ha").isFloat({ min: 0.01 }),
    body("phone").optional().isString(),
  ],
  validate,
  (req, res) => {
    try {
      const farmer = dataService.createFarmer(req.body);
      logger.info(`New farmer registered: ${farmer.name} in ${farmer.region}`);
      res.status(201).json({ success: true, farmer });
    } catch (err) {
      res.status(500).json({ success: false, error: err.message });
    }
  }
);

// ── GET /api/farmers/:id ─────────────────────────────────────────────────────
router.get("/:id", (req, res) => {
  const farmer = dataService.getFarmerById(req.params.id);
  if (!farmer) {
    return res.status(404).json({ success: false, error: "Farmer not found." });
  }
  res.json({ success: true, farmer });
});

// ── GET /api/farmers/:id/history ──────────────────────────────────────────────
router.get(
  "/:id/history",
  [param("id").isString(), query("limit").optional().isInt({ min: 1, max: 100 })],
  validate,
  (req, res) => {
    const farmer = dataService.getFarmerById(req.params.id);
    if (!farmer) {
      return res.status(404).json({ success: false, error: "Farmer not found." });
    }
    const limit   = parseInt(req.query.limit) || 20;
    const records = dataService.getFarmerHistory(req.params.id, limit);
    res.json({
      success: true,
      farmer_name: farmer.name,
      region: farmer.region,
      count: records.length,
      history: records,
    });
  }
);

// ── GET /api/farmers/history/all ─────────────────────────────────────────────
router.get("/history/all", (req, res) => {
  const limit   = parseInt(req.query.limit) || 50;
  const records = dataService.getAllHistory(limit);
  res.json({ success: true, count: records.length, history: records });
});

module.exports = router;
