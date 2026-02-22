// src/routes/data.js
// Static reference data: Rwanda regions, crop types, soil types, seasons.
// Also provides a sensor-simulation endpoint for testing.

const express = require("express");
const router  = express.Router();

const REGIONS = [
  { id: "kigali",   name: "Kigali",   districts: ["Gasabo", "Kicukiro", "Nyarugenge"] },
  { id: "northern", name: "Northern", districts: ["Burera", "Gakenke", "Gicumbi", "Musanze", "Rulindo"] },
  { id: "southern", name: "Southern", districts: ["Gisagara", "Huye", "Kamonyi", "Muhanga", "Nyamagabe", "Nyanza", "Nyaruguru", "Ruhango"] },
  { id: "eastern",  name: "Eastern",  districts: ["Bugesera", "Gatsibo", "Kayonza", "Kirehe", "Ngoma", "Nyagatare", "Rwamagana"] },
  { id: "western",  name: "Western",  districts: ["Karongi", "Ngororero", "Nyabihu", "Nyamasheke", "Rubavu", "Rusizi", "Rutsiro"] },
];

const CROPS = [
  { id: "maize",       name: "Maize",       seasons: ["rainy_A", "rainy_B"], avg_yield_kg_ha: 2000 },
  { id: "beans",       name: "Beans",       seasons: ["rainy_A", "rainy_B"], avg_yield_kg_ha: 1200 },
  { id: "sorghum",     name: "Sorghum",     seasons: ["rainy_A"],            avg_yield_kg_ha: 1500 },
  { id: "cassava",     name: "Cassava",     seasons: ["rainy_A", "rainy_B"], avg_yield_kg_ha: 8000 },
  { id: "sweet_potato",name: "Sweet Potato",seasons: ["rainy_A", "rainy_B"], avg_yield_kg_ha: 6000 },
];

const SOIL_TYPES = [
  { id: "clay",     name: "Clay",           water_retention: "high",   fertility: "medium" },
  { id: "loam",     name: "Loam",           water_retention: "medium", fertility: "high"   },
  { id: "sandy",    name: "Sandy",          water_retention: "low",    fertility: "low"    },
  { id: "volcanic", name: "Volcanic (Andosol)", water_retention: "medium", fertility: "very_high" },
];

const SEASONS = [
  { id: "rainy_A", name: "Season A", months: "March–May",       rainfall: "High" },
  { id: "rainy_B", name: "Season B", months: "September–November", rainfall: "Moderate" },
  { id: "dry",     name: "Dry Season", months: "June–August, December–February", rainfall: "Low" },
];

// ── GET /api/data/regions ─────────────────────────────────────────────────────
router.get("/regions", (req, res) => {
  res.json({ success: true, count: REGIONS.length, regions: REGIONS });
});

// ── GET /api/data/crops ───────────────────────────────────────────────────────
router.get("/crops", (req, res) => {
  res.json({ success: true, count: CROPS.length, crops: CROPS });
});

// ── GET /api/data/soils ───────────────────────────────────────────────────────
router.get("/soils", (req, res) => {
  res.json({ success: true, count: SOIL_TYPES.length, soil_types: SOIL_TYPES });
});

// ── GET /api/data/seasons ─────────────────────────────────────────────────────
router.get("/seasons", (req, res) => {
  res.json({ success: true, seasons: SEASONS });
});

// ── GET /api/data/simulate ────────────────────────────────────────────────────
// Simulate a sensor/satellite reading for a given month.
// Use for testing the prediction pipeline.
router.get("/simulate", (req, res) => {
  const month     = parseInt(req.query.month) || new Date().getMonth() + 1;
  const regionIdx = parseInt(req.query.region_index) || 0;
  const regions   = ["Kigali", "Northern", "Southern", "Eastern", "Western"];
  const crops     = ["maize", "beans", "sorghum", "cassava", "sweet_potato"];
  const soils     = ["clay", "loam", "sandy", "volcanic"];

  const rainMap = { 1:50,2:70,3:130,4:160,5:120,6:30,7:15,8:20,9:80,10:110,11:120,12:60 };
  const seasonMap = { 1:"dry",2:"dry",3:"rainy_A",4:"rainy_A",5:"rainy_A",6:"dry",7:"dry",8:"dry",9:"rainy_B",10:"rainy_B",11:"rainy_B",12:"dry" };

  const rainfall   = Math.max(0, rainMap[month] + (Math.random() * 30 - 15));
  const simulated  = {
    region:                  regions[regionIdx % regions.length],
    season:                  seasonMap[month],
    crop_type:               crops[Math.floor(Math.random() * crops.length)],
    soil_type:               soils[Math.floor(Math.random() * soils.length)],
    temperature_c:           parseFloat((18 + Math.random() * 6).toFixed(2)),
    humidity_pct:            parseFloat((50 + rainfall * 0.2 + Math.random() * 10).toFixed(2)),
    rainfall_mm:             parseFloat(rainfall.toFixed(2)),
    soil_moisture_pct:       parseFloat((20 + rainfall * 0.4 + Math.random() * 10).toFixed(2)),
    wind_speed_kmh:          parseFloat((8 + Math.random() * 8).toFixed(2)),
    solar_radiation_wm2:     parseFloat((180 + Math.random() * 60).toFixed(2)),
    ndvi:                    parseFloat((0.3 + Math.random() * 0.5).toFixed(4)),
    pest_pressure_index:     parseFloat((Math.random() * 0.5).toFixed(4)),
    fertilizer_applied_kg_ha:parseFloat([0, 25, 50, 75][Math.floor(Math.random() * 4)]),
    irrigation_applied_mm:   parseFloat((rainfall < 20 ? Math.random() * 10 : 0).toFixed(2)),
    drought_index:           parseFloat(Math.max(0, Math.min(1, 1 - rainfall / 160 + (Math.random() * 0.1 - 0.05))).toFixed(4)),
  };

  res.json({
    success: true,
    simulated_for: { month, region: simulated.region, season: simulated.season },
    record: simulated,
    next_step: "POST this record to /api/predict/full to get AI advice.",
  });
});

module.exports = router;
