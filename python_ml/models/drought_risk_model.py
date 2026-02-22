"""
drought_risk_model.py
Gradient Boosting + PyTorch regression → continuous drought_index [0, 1].
Also contains a rule-based Drought Risk Logic layer that converts the
numeric score into an actionable risk category.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.data_processor import prepare_split, save_artifacts, preprocess_single

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
MODEL_NAME = "drought_risk"

# ── Risk thresholds ───────────────────────────────────────────────────────────
DROUGHT_LEVELS = {
    (0.0, 0.20): ("none",     "No drought stress detected."),
    (0.20, 0.40): ("mild",    "Mild dryness — monitor soil moisture closely."),
    (0.40, 0.60): ("moderate","Moderate drought — consider supplemental irrigation."),
    (0.60, 0.80): ("severe",  "Severe drought — immediate water conservation needed."),
    (0.80, 1.01): ("extreme", "EXTREME drought — crop loss risk is high. Emergency action required."),
}


def classify_drought(score: float) -> dict:
    for (lo, hi), (level, message) in DROUGHT_LEVELS.items():
        if lo <= score < hi:
            return {"level": level, "score": round(score, 4), "message": message}
    return {"level": "extreme", "score": round(score, 4), "message": "EXTREME drought."}


# ── Gradient Boosting Regressor ───────────────────────────────────────────────

class DroughtGBRegressor:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        )
        self.trained = False

    def train(self, X_tr, y_tr):
        print("[DroughtGB] Training Gradient Boosting …")
        self.model.fit(X_tr, y_tr)
        self.trained = True
        print("[DroughtGB] Done.")

    def predict(self, X) -> np.ndarray:
        return np.clip(self.model.predict(X), 0, 1)

    def evaluate(self, X_te, y_te):
        preds = self.predict(X_te)
        print("\n── Drought Regressor Evaluation ────────────────────")
        print(f"MAE : {mean_absolute_error(y_te, preds):.4f}")
        print(f"R²  : {r2_score(y_te, preds):.4f}")

    def save(self):
        path = os.path.join(ARTIFACTS_DIR, f"{MODEL_NAME}_gb.pkl")
        joblib.dump(self.model, path)
        print(f"[DroughtGB] Saved → {path}")

    def load(self):
        path = os.path.join(ARTIFACTS_DIR, f"{MODEL_NAME}_gb.pkl")
        self.model = joblib.load(path)
        self.trained = True


# ── PyTorch LSTM for time-aware drought tracking ──────────────────────────────

class DroughtLSTM(nn.Module):
    """Seq-to-one LSTM: uses a rolling window of climate features."""

    def __init__(self, input_dim: int, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers,
                            batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):                     # x: (B, T, F)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class DroughtLSTMTrainer:
    WINDOW = 7   # days of history

    def __init__(self, input_dim: int, device: str = "cpu"):
        self.device = torch.device(device)
        self.input_dim = input_dim
        self.model = DroughtLSTM(input_dim).to(self.device)
        self.trained = False

    def _make_sequences(self, X: np.ndarray, y: np.ndarray):
        Xs, ys = [], []
        for i in range(self.WINDOW, len(X)):
            Xs.append(X[i - self.WINDOW: i])
            ys.append(y[i])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    def train(self, X_tr, y_tr, epochs: int = 25, lr: float = 1e-3, batch: int = 128):
        Xs, ys = self._make_sequences(X_tr, y_tr)
        ds = torch.utils.data.TensorDataset(
            torch.tensor(Xs), torch.tensor(ys)
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        print(f"[DroughtLSTM] Training for {epochs} epochs (window={self.WINDOW}) …")
        for ep in range(1, epochs + 1):
            self.model.train()
            total = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                total += loss.item()
            if ep % 5 == 0:
                print(f"  Epoch {ep:3d} | MSE: {total/len(loader):.5f}")
        self.trained = True

    def predict(self, X: np.ndarray) -> float:
        """Predict drought index for the latest sample (requires WINDOW rows)."""
        if len(X) < self.WINDOW:
            raise ValueError(f"Need at least {self.WINDOW} rows for LSTM prediction.")
        seq = torch.tensor(
            X[-self.WINDOW:][np.newaxis], dtype=torch.float32
        ).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return float(np.clip(self.model(seq).item(), 0, 1))

    def save(self):
        path = os.path.join(ARTIFACTS_DIR, f"{MODEL_NAME}_lstm.pt")
        torch.save({
            "state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
        }, path)
        print(f"[DroughtLSTM] Saved → {path}")

    def load(self):
        path = os.path.join(ARTIFACTS_DIR, f"{MODEL_NAME}_lstm.pt")
        ckpt = torch.load(path, map_location="cpu")
        self.model = DroughtLSTM(ckpt["input_dim"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.trained = True


# ── full train pipeline ───────────────────────────────────────────────────────

def train_drought_model(df):
    print("\n══ DROUGHT RISK MODEL ═══════════════════════════════")

    # exclude drought_index itself from features
    X_tr, X_te, y_tr, y_te, meta = prepare_split(df, "drought_index")

    # Gradient Boosting
    gb = DroughtGBRegressor()
    gb.train(X_tr, y_tr)
    gb.evaluate(X_te, y_te)
    gb.save()

    # LSTM (trained on full training set ordered by time)
    lstm = DroughtLSTMTrainer(input_dim=X_tr.shape[1])
    lstm.train(X_tr, y_tr, epochs=25)
    lstm.save()

    save_artifacts(MODEL_NAME, meta)
    return gb, lstm, meta
