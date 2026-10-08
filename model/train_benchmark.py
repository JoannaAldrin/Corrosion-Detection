"""Benchmarks the physics-informed LSTM against the plain baseline LSTM,
reporting RMSE separately for external vs internal corrosion."""
import json, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from common import build_windows, CorrosionSequenceDataset, CorrosionLSTM, ALL_FEATURES

df = pd.read_csv("data/simulated_corrosion.csv")
pipe_ids = sorted(df["pipe_id"].unique())
test_pipes = set(pipe_ids[::5])
train_df = df[~df["pipe_id"].isin(test_pipes)].copy()
test_df = df[df["pipe_id"].isin(test_pipes)].copy()

means, stds = train_df[ALL_FEATURES].mean(), train_df[ALL_FEATURES].std()
for d in (train_df, test_df):
    d[ALL_FEATURES] = (d[ALL_FEATURES] - means) / stds

X_test, y_test = build_windows(test_df)
test_loader = DataLoader(CorrosionSequenceDataset(X_test, y_test), batch_size=64, shuffle=False)

def evaluate(model_path):
    model = CorrosionLSTM()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.append(model(xb).numpy())
            actuals.append(yb.numpy())
    preds, actuals = np.concatenate(preds), np.concatenate(actuals)
    rmse_ext = float(np.sqrt(np.mean((preds[:, 0] - actuals[:, 0]) ** 2)))
    rmse_int = float(np.sqrt(np.mean((preds[:, 1] - actuals[:, 1]) ** 2)))
    return rmse_ext, rmse_int

base_ext, base_int = evaluate("model/baseline_lstm_model.pth")
pinn_ext, pinn_int = evaluate("model/pinn_lstm_model.pth")

results = {
    "baseline": {"rmse_external": base_ext, "rmse_internal": base_int},
    "physics_informed": {"rmse_external": pinn_ext, "rmse_internal": pinn_int},
    "improvement_pct_external": round(100 * (base_ext - pinn_ext) / base_ext, 2),
}
with open("model/benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))