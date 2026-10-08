"""Shared dataset windowing and model architecture — import this from
baseline_lstm.ipynb, physics_informed_lstm.ipynb, train_shap_model.py,
and train_benchmark.py so all four stay consistent."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

EXTERNAL_FEATURES = ["temperature_C", "humidity_pct", "salinity_pct", "ph"]
INTERNAL_FEATURES = ["internal_temp_C", "water_cut_pct", "flow_velocity_m_s"]
ALL_FEATURES = EXTERNAL_FEATURES + INTERNAL_FEATURES
TARGETS = ["external_corrosion_depth_mm", "internal_corrosion_depth_mm"]
WINDOW = 30

def build_windows(df, feature_cols=ALL_FEATURES, target_cols=TARGETS, window=WINDOW):
    X, y = [], []
    for pipe_id, group in df.groupby("pipe_id"):
        group = group.sort_values("day")
        feats = group[feature_cols].values
        targets = group[target_cols].values
        for i in range(len(group) - window):
            X.append(feats[i:i + window])
            y.append(targets[i + window])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

class CorrosionSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CorrosionLSTM(nn.Module):
    """One shared backbone, two output heads — external and internal
    corrosion are physically distinct processes, so they get separate
    prediction heads instead of being blended into one number."""
    def __init__(self, input_size=len(ALL_FEATURES), hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.external_head = nn.Linear(hidden_size, 1)
        self.internal_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return torch.cat([self.external_head(last), self.internal_head(last)], dim=-1)