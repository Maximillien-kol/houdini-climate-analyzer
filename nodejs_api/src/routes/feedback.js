// src/routes/feedback.js
// Submit real-world farmer outcomes to the AI system.
// This powers the "Track Results → Learn" loop.

const express = require("express");
const { body, validationResult } = require("express-validator");
const mlService = require("../services/mlService");
const logger    = require("../utils/logger");

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

// ── POST /api/feedback ────────────────────────────────────────────────────────
// After a farmer follows advice, log what actually happened.
// This data feeds future model retraining.
router.post(
  "/",
  [
    body("feedback_id").isString().notEmpty().withMessage("feedback_id from the prediction report is required"),
    body("actual_outcome").isObject().withMessage("actual_outcome must be a JSON object"),
    body("actual_outcome.rained").optional().isBoolean(),
    body("actual_outcome.actual_yield_kg_ha").optional().isFloat({ min: 0 }),
    body("actual_outcome.crop_health_observed").optional().isFloat({ min: 0, max: 100 }),
    body("actual_outcome.helpful_advice").optional().isArray(),
  ],
  validate,
  async (req, res) => {
    try {
      const { feedback_id, actual_outcome } = req.body;
      const result = await mlService.submitFeedback(feedback_id, actual_outcome);

      logger.info(`Feedback logged: ${feedback_id}`);
      res.json({
        success: true,
        message: "Thank you! Your feedback helps the AI learn and improve.",
        logged: result,
      });
    } catch (err) {
      logger.error(`/feedback: ${err.message}`);
      res.status(502).json({ success: false, error: err.message });
    }
  }
);

// ── GET /api/feedback/summary ─────────────────────────────────────────────────
// Model drift monitoring — compare predictions versus actuals.
router.get("/summary", async (req, res) => {
  try {
    const summary = await mlService.getFeedbackSummary();
    res.json({ success: true, drift_report: summary });
  } catch (err) {
    res.status(502).json({ success: false, error: err.message });
  }
});

module.exports = router;
