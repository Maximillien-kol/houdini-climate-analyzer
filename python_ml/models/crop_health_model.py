"""
crop_health_model.py
Deep Neural Network (PyTorch) → predicts crop_health_score [0-100]
and yield_kg_ha. Also handles pest-pressure risk classification.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.data_processor import prepare_split, save_artifacts

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
MODEL_NAME = "crop_health"

PEST_THRESHOLDS = {
    (0.0, 0.25): ("low",      "Pest pressure is low — routine monitoring sufficient."),
    (0.25, 0.55): ("medium",  "Moderate pest pressure — inspect crops weekly."),
    (0.55, 0.80): ("high",    "High pest pressure — apply biological or chemical control."),
    (0.80, 1.01): ("critical","Critical pest outbreak — immediate intervention required."),
}


def classify_pest(index: float) -> dict:
    for (lo, hi), (level, message) in PEST_THRESHOLDS.items():
        if lo <= index < hi:
            return {"level": level, "pest_pressure_index": round(index, 4), "message": message}
    return {"level": "critical", "pest_pressure_index": round(index, 4), "message": "Critical."}


# ── Multi-output DNN ──────────────────────────────────────────────────────────

class CropHealthDNN(nn.Module):
    """
    Shared backbone → two heads:
      head_health : crop_health_score  [0,100]
      head_yield  : yield_kg_ha        [0, ∞)
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        # health score head (sigmoid * 100)
        self.head_health = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        # yield head (ReLU → no negative yields)
        self.head_yield = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.ReLU()
        )

    def forward(self, x):
        feat = self.backbone(x)
        health = self.head_health(feat).squeeze(-1) * 100  # 0–100
        yield_ = self.head_yield(feat).squeeze(-1) * 3000  # up to 3000 kg/ha
        return health, yield_


class CropHealthTrainer:
    YIELD_SCALE = 3000.0

    def __init__(self, input_dim: int, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = CropHealthDNN(input_dim).to(self.device)
        self.input_dim = input_dim
        self.trained = False

    def _tensors(self, X, y_health, y_yield):
        return (
            torch.tensor(X, dtype=torch.float32).to(self.device),
            torch.tensor(y_health, dtype=torch.float32).to(self.device),
            torch.tensor(y_yield, dtype=torch.float32).to(self.device),
        )

    def train(self, X_tr, y_health_tr, y_yield_tr,
              epochs: int = 40, lr: float = 1e-3, batch: int = 256):
        X, yh, yy = self._tensors(X_tr, y_health_tr, y_yield_tr)
        ds = torch.utils.data.TensorDataset(X, yh, yy)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        mse = nn.MSELoss()

        print(f"[CropDNN] Training multi-output DNN for {epochs} epochs …")
        for ep in range(1, epochs + 1):
            self.model.train()
            total = 0
            for xb, yhb, yyb in loader:
                opt.zero_grad()
                h_pred, y_pred = self.model(xb)
                # normalise yield into [0,1] for balanced loss
                loss = mse(h_pred / 100, yhb / 100) + mse(y_pred / self.YIELD_SCALE,
                                                           yyb / self.YIELD_SCALE)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                total += loss.item()
            scheduler.step()
            if ep % 10 == 0:
                print(f"  Epoch {ep:3d} | Loss: {total/len(loader):.5f}")
        self.trained = True

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        t = torch.tensor(X, dtype=torch.float32).to(self.device)
        h, y = self.model(t)
        return h.cpu().numpy(), y.cpu().numpy()

    def evaluate(self, X_te, y_health_te, y_yield_te):
        h_pred, y_pred = self.predict(X_te)
        print("\n── Crop Health DNN Evaluation ──────────────────────")
        print(f"Health MAE : {mean_absolute_error(y_health_te, h_pred):.2f}")
        print(f"Health R²  : {r2_score(y_health_te, h_pred):.4f}")
        print(f"Yield  MAE : {mean_absolute_error(y_yield_te,  y_pred):.2f} kg/ha")
        print(f"Yield  R²  : {r2_score(y_yield_te,  y_pred):.4f}")

    def save(self):
        path = os.path.join(ARTIFACTS_DIR, f"{MODEL_NAME}_dnn.pt")
        torch.save({
            "state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
        }, path)
        print(f"[CropDNN] Saved → {path}")

    def load(self):
        path = os.path.join(ARTIFACTS_DIR, f"{MODEL_NAME}_dnn.pt")
        ckpt = torch.load(path, map_location="cpu")
        self.model = CropHealthDNN(ckpt["input_dim"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.trained = True


# ── full train pipeline ───────────────────────────────────────────────────────

def train_crop_model(df):
    print("\n══ CROP HEALTH MODEL ════════════════════════════════")

    from sklearn.model_selection import train_test_split
    from utils.data_processor import encode_categoricals, build_feature_matrix, scale_features

    df2, encoders = encode_categoricals(df.copy(), fit=True)
    X = build_feature_matrix(df2)
    y_health = df2["crop_health_score"].values.astype(np.float32)
    y_yield  = df2["yield_kg_ha"].values.astype(np.float32)

    X_tr, X_te, yh_tr, yh_te, yy_tr, yy_te = _multi_split(
        X, y_health, y_yield
    )
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    trainer = CropHealthTrainer(input_dim=X_tr.shape[1])
    trainer.train(X_tr, yh_tr, yy_tr, epochs=40)
    trainer.evaluate(X_te, yh_te, yy_te)
    trainer.save()

    meta = {"encoders": encoders, "scaler": scaler}
    save_artifacts(MODEL_NAME, meta)
    return trainer, meta


def _multi_split(X, y1, y2, test_size=0.2):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(X))
    idx_tr, idx_te = train_test_split(idx, test_size=test_size, random_state=42)
    return (X[idx_tr], X[idx_te],
            y1[idx_tr], y1[idx_te],
            y2[idx_tr], y2[idx_te])
