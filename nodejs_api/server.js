// server.js — AgriShield AI Node.js API Gateway
// Starts the Express server and manages Python ML service lifecycle.

require("dotenv").config();
const app    = require("./src/app");
const logger = require("./src/utils/logger");

const PORT = process.env.PORT || 3000;

const server = app.listen(PORT, () => {
  logger.info(`╔══════════════════════════════════════════════════════╗`);
  logger.info(`║  AgriShield AI — Node.js API Gateway                 ║`);
  logger.info(`║  Food Insecurity & Climate Vulnerability System       ║`);
  logger.info(`╚══════════════════════════════════════════════════════╝`);
  logger.info(`Server running on http://localhost:${PORT}`);
  logger.info(`ML Service: ${process.env.ML_SERVICE_URL || "http://localhost:5001"}`);
  logger.info(`Endpoints:`);
  logger.info(`  GET  /api/health`);
  logger.info(`  POST /api/predict/rain`);
  logger.info(`  POST /api/predict/drought`);
  logger.info(`  POST /api/predict/crop`);
  logger.info(`  POST /api/predict/full`);
  logger.info(`  GET  /api/farmers`);
  logger.info(`  POST /api/farmers`);
  logger.info(`  GET  /api/farmers/:id/history`);
  logger.info(`  POST /api/feedback`);
  logger.info(`  GET  /api/feedback/summary`);
  logger.info(`  GET  /api/data/regions`);
  logger.info(`  GET  /api/data/crops`);
});

// Graceful shutdown
process.on("SIGTERM", () => {
  logger.info("SIGTERM received — shutting down gracefully.");
  server.close(() => {
    logger.info("HTTP server closed.");
    process.exit(0);
  });
});
