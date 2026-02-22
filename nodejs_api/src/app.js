// src/app.js - Express application configuration

const express     = require("express");
const cors        = require("cors");
const helmet      = require("helmet");
const compression = require("compression");
const morgan      = require("morgan");
const rateLimit   = require("express-rate-limit");

const predictionsRouter = require("./routes/predictions");
const farmersRouter     = require("./routes/farmers");
const dataRouter        = require("./routes/data");
const feedbackRouter    = require("./routes/feedback");
const logger            = require("./utils/logger");

const app = express();

// ── Security & compression ────────────────────────────────────────────────────
app.use(helmet());
app.use(compression());
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(",") || "*",
  methods: ["GET", "POST", "PUT", "DELETE"],
}));

// ── Rate limiting ─────────────────────────────────────────────────────────────
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,   // 15 minutes
  max: 200,
  standardHeaders: true,
  legacyHeaders: false,
  message: { success: false, error: "Too many requests - please try again later." },
});
app.use("/api/", limiter);

// ── Body parsing ──────────────────────────────────────────────────────────────
app.use(express.json({ limit: "1mb" }));
app.use(express.urlencoded({ extended: true }));

// ── HTTP logging ──────────────────────────────────────────────────────────────
app.use(morgan("combined", {
  stream: { write: (msg) => logger.http(msg.trim()) },
}));

// ── Routes ────────────────────────────────────────────────────────────────────
app.use("/api/predict",  predictionsRouter);
app.use("/api/farmers",  farmersRouter);
app.use("/api/data",      dataRouter);
app.use("/api/feedback",  feedbackRouter);

// ── Root ──────────────────────────────────────────────────────────────────────
app.get("/", (req, res) => {
  res.json({
    service: "Rwac V.0.1 - Food Insecurity & Climate Vulnerability API",
    version: "1.0.0",
    status: "running",
    docs: "See README.md for full API documentation.",
  });
});

app.get("/api/health", (req, res) => {
  res.json({
    success: true,
    status: "healthy",
    uptime_seconds: Math.floor(process.uptime()),
    memory_mb: Math.round(process.memoryUsage().heapUsed / 1024 / 1024),
    timestamp: new Date().toISOString(),
  });
});

// ── 404 ───────────────────────────────────────────────────────────────────────
app.use((req, res) => {
  res.status(404).json({ success: false, error: `Route ${req.path} not found.` });
});

// ── Global error handler ──────────────────────────────────────────────────────
app.use((err, req, res, next) => {
  logger.error(`Unhandled error: ${err.message}`);
  res.status(500).json({ success: false, error: "Internal server error." });
});

module.exports = app;
