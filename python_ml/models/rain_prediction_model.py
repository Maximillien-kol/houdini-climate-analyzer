"""
rain_prediction_model.py
Random Forest classifier → predicts rain_tomorrow (0/1).
Uses scikit-learn + a thin PyTorch wrapper for probability calibration.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score
)
from sklearn.calibration import CalibratedClassifierCV

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.data_processor import prepare_split, save_artifacts, load_artifacts, preprocess_single

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
MODEL_NAME = "rain_prediction"


# ── Random Forest (primary model) ────────────────────────────────────────────

class RainRandomForest:
    """Calibrated Random Forest for next-day rain probability."""

    def __init__(self):
        base = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.model = CalibratedClassifierCV(base, cv=3, method="sigmoid")
        self.trained = False

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray):
        print("[RainRF] Training Random Forest + calibration …")
        self.model.fit(X_tr, y_tr.astype(int))
        self.trained = True
        print("[RainRF] Training complete.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def evaluate(self, X_te: np.ndarray, y_te: np.ndarray):
        proba = self.predict_proba(X_te)
        preds = (proba >= 0.5).astype(int)
        print("\n── Rain Prediction Evaluation ─────────────────────")
        print(classification_report(y_te.astype(int), preds,
                                     target_names=["No Rain", "Rain"]))
        print(f"AUC-ROC : {roc_auc_score(y_te, proba):.4f}")
        print(f"Accuracy: {accuracy_score(y_te, preds):.4f}")

    def save(self):
        path = os.path.join(ARTIFACTS_DIR, f"{MODEL_NAME}_rf.pkl")
        joblib.dump(self.model, path)
        print(f"[RainRF] Saved → {path}")

    def load(self):
        path = os.path.join(ARTIFACTS_DIR, f"{MODEL_NAME}_rf.pkl")
        self.model = joblib.load(path)
        self.trained = True


# ── PyTorch MLP (secondary / ensemble) ───────────────────────────────────────

class RainMLP(nn.Module):
    """Lightweight MLP for rain classification (used to ensemble with RF)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RainMLPTrainer:
    def __init__(self, input_dim: int, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = RainMLP(input_dim).to(self.device)
        self.trained = False

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray,
              epochs: int = 30, lr: float = 1e-3, batch: int = 256):
        X = torch.tensor(X_tr, dtype=torch.float32).to(self.device)
        y = torch.tensor(y_tr, dtype=torch.float32).to(self.device)

        ds = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

        print(f"[RainMLP] Training for {epochs} epochs …")
        for ep in range(1, epochs + 1):
            self.model.train()
            total_loss = 0
            for xb, yb in loader:
                opt.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            scheduler.step()
            if ep % 10 == 0:
                print(f"  Epoch {ep:3d} | Loss: {total_loss/len(loader):.4f}")
        self.trained = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.model(t).cpu().numpy()

    def save(self):
        path = os.path.join(ARTIFACTS_DIR, f"{MODEL_NAME}_mlp.pt")
        torch.save(self.model.state_dict(), path)
        print(f"[RainMLP] Saved → {path}")

    def load(self, input_dim: int):
        path = os.path.join(ARTIFACTS_DIR, f"{MODEL_NAME}_mlp.pt")
        self.model = RainMLP(input_dim)
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.model.eval()
        self.trained = True


# ── Ensemble ──────────────────────────────────────────────────────────────────

def ensemble_predict(rf: RainRandomForest, mlp: RainMLPTrainer,
                     X: np.ndarray, threshold: float = 0.5) -> dict:
    p_rf = rf.predict_proba(X)
    p_mlp = mlp.predict_proba(X)
    p_ensemble = 0.6 * p_rf + 0.4 * p_mlp
    label = int(p_ensemble[0] >= threshold)
    return {
        "rain_tomorrow": bool(label),
        "probability": float(round(p_ensemble[0], 4)),
        "rf_probability": float(round(p_rf[0], 4)),
        "mlp_probability": float(round(p_mlp[0], 4)),
    }


# ── full train pipeline ───────────────────────────────────────────────────────

def train_rain_model(df):
    print("\n══ RAIN PREDICTION MODEL ════════════════════════════")
    X_tr, X_te, y_tr, y_te, meta = prepare_split(df, "rain_tomorrow")

    # Random Forest
    rf = RainRandomForest()
    rf.train(X_tr, y_tr)
    rf.evaluate(X_te, y_te)
    rf.save()

    # PyTorch MLP
    mlp = RainMLPTrainer(input_dim=X_tr.shape[1])
    mlp.train(X_tr, y_tr, epochs=30)
    mlp.save()

    save_artifacts(MODEL_NAME, meta)
    print(f"[Rain] Input dim: {X_tr.shape[1]}")
    return rf, mlp, meta
