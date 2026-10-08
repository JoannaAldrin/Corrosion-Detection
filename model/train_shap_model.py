"""
Explainability for the two-headed physics-informed LSTM.
Note: TreeExplainer (used for the old XGBoost corrosion_model.pkl) does
NOT work on this PyTorch model — GradientExplainer is required instead.
"""
import torch, shap, numpy as np, pandas as pd, joblib
from common import build_windows, ALL_FEATURES, CorrosionLSTM

df = pd.read_csv("data/simulated_corrosion.csv")
means, stds = df[ALL_FEATURES].mean(), df[ALL_FEATURES].std()
df[ALL_FEATURES] = (df[ALL_FEATURES] - means) / stds
X, y = build_windows(df)

model = CorrosionLSTM()
model.load_state_dict(torch.load("model/pinn_lstm_model.pth", map_location="cpu"))
model.eval()

background = torch.from_numpy(X[np.random.choice(len(X), 50, replace=False)])
sample = torch.from_numpy(X[:200])

explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(sample)  # list: [external_output, internal_output]

external_importance = np.abs(shap_values[0]).mean(axis=(0, 1))
internal_importance = np.abs(shap_values[1]).mean(axis=(0, 1))

print("External corrosion — feature attribution:")
for name, val in zip(ALL_FEATURES, external_importance):
    print(f"  {name}: {val:.4f}")
print("Internal (MIC) corrosion — feature attribution:")
for name, val in zip(ALL_FEATURES, internal_importance):
    print(f"  {name}: {val:.4f}")

joblib.dump({"explainer": explainer, "feature_names": ALL_FEATURES},
            "model/shap_explainer.pkl")