// src/services/dataService.js
// In-memory farmer database + prediction history tracker.
// In production, replace with PostgreSQL / MongoDB.

const { v4: uuidv4 } = require("uuid");
const logger = require("../utils/logger");

// ── In-memory stores ──────────────────────────────────────────────────────────
const farmers  = new Map();   // id → farmer object
const history  = new Map();   // farmerId → [ prediction records ]

// ── Seed data — sample Rwandan farmers ───────────────────────────────────────
function seedFarmers() {
  const samples = [
    {
      name: "Jean-Pierre Nkurunziza",
      region: "Eastern",
      district: "Kayonza",
      crop_type: "maize",
      soil_type: "loam",
      farm_size_ha: 1.5,
      phone: "+250788000001",
    },
    {
      name: "Marie Uwase",
      region: "Southern",
      district: "Huye",
      crop_type: "beans",
      soil_type: "volcanic",
      farm_size_ha: 0.8,
      phone: "+250788000002",
    },
    {
      name: "Emmanuel Habimana",
      region: "Northern",
      district: "Musanze",
      crop_type: "sweet_potato",
      soil_type: "volcanic",
      farm_size_ha: 2.0,
      phone: "+250788000003",
    },
    {
      name: "Claudine Mukamana",
      region: "Western",
      district: "Rusizi",
      crop_type: "cassava",
      soil_type: "sandy",
      farm_size_ha: 1.2,
      phone: "+250788000004",
    },
  ];

  samples.forEach((f) => {
    const id = uuidv4();
    farmers.set(id, { id, ...f, registered_at: new Date().toISOString() });
    history.set(id, []);
  });
  logger.info(`[DataService] Seeded ${samples.length} sample farmers.`);
}

seedFarmers();

// ── Farmer CRUD ───────────────────────────────────────────────────────────────

function createFarmer(data) {
  const id = uuidv4();
  const farmer = { id, ...data, registered_at: new Date().toISOString() };
  farmers.set(id, farmer);
  history.set(id, []);
  logger.info(`[DataService] Farmer created: ${farmer.name} (${id})`);
  return farmer;
}

function getAllFarmers() {
  return [...farmers.values()];
}

function getFarmerById(id) {
  return farmers.get(id) || null;
}

function updateFarmer(id, updates) {
  const farmer = farmers.get(id);
  if (!farmer) return null;
  const updated = { ...farmer, ...updates, updated_at: new Date().toISOString() };
  farmers.set(id, updated);
  return updated;
}

// ── Prediction history ────────────────────────────────────────────────────────

function savePrediction(farmerId, prediction) {
  const record = {
    id: uuidv4(),
    farmer_id: farmerId,
    timestamp: new Date().toISOString(),
    ...prediction,
  };

  if (!history.has(farmerId)) {
    history.set(farmerId, []);
  }
  const farmerHistory = history.get(farmerId);
  farmerHistory.unshift(record);           // newest first
  if (farmerHistory.length > 100) {        // cap at 100 records per farmer
    farmerHistory.pop();
  }
  return record;
}

function getFarmerHistory(farmerId, limit = 20) {
  return (history.get(farmerId) || []).slice(0, limit);
}

function getAllHistory(limit = 50) {
  const all = [];
  for (const records of history.values()) {
    all.push(...records);
  }
  return all
    .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
    .slice(0, limit);
}

// ── Stats ─────────────────────────────────────────────────────────────────────

function getSystemStats() {
  let totalPredictions = 0;
  for (const records of history.values()) {
    totalPredictions += records.length;
  }
  return {
    farmers_registered: farmers.size,
    total_predictions: totalPredictions,
    regions_covered: [...new Set([...farmers.values()].map((f) => f.region))],
  };
}

module.exports = {
  createFarmer,
  getAllFarmers,
  getFarmerById,
  updateFarmer,
  savePrediction,
  getFarmerHistory,
  getAllHistory,
  getSystemStats,
};
