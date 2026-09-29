import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Load Telemetry Data
df = pd.read_csv("data/simulated_corrosion.csv")
features = ['temperature_C', 'humidity_pct', 'salinity_gL', 'srb_multiplier']
target = 'corrosion_rate'

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(df[features].values)
y_scaled = scaler_y.fit_transform(df[[target]].values)

# Create 24-hour sequence windows
def create_sequences(X, y, seq_len=24):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled)

# Chronological Train/Test Split (75% Train, 25% Test)
split_idx = int(len(X_seq) * 0.75)
X_train = torch.tensor(X_seq[:split_idx], dtype=torch.float32)
y_train = torch.tensor(y_seq[:split_idx], dtype=torch.float32)
X_test = torch.tensor(X_seq[split_idx:], dtype=torch.float32)
y_test = torch.tensor(y_seq[split_idx:], dtype=torch.float32)

# 2. LSTM Architecture
class PipelineLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# 3. Physics-Informed Loss Function (Normalized Scale)
def pinn_loss_function(y_pred, y_true, X_batch, epoch, max_epochs):
    # Core Data-Driven MSE Loss
    mse = nn.MSELoss()(y_pred, y_true)
    
    # Annealing factor: gradually turn on physics penalties over training
    anneal = min(1.0, epoch / (max_epochs * 0.5))
    
    # Physics Penalty 1: Soft Non-Negativity (corrosion rate cannot be < 0)
    # Convert prediction back to physical scale approximation for penalty calculation
    neg_penalty = torch.mean(torch.square(torch.relu(-y_pred)))
    
    # Physics Penalty 2: Monotonicity with respect to high SRB microbial bloom
    # If SRB multiplier (col index 3 in X_batch) > 2.0 (scaled value > ~1.0), penalize under-predictions
    srb_high_mask = (X_batch[:, -1, 3] > 1.0).unsqueeze(1)
    srb_underpredict_penalty = torch.mean(torch.square(torch.relu(y_true - y_pred) * srb_high_mask))
    
    total_loss = mse + anneal * (0.01 * neg_penalty + 0.05 * srb_underpredict_penalty)
    return total_loss

# 4. Training & Benchmarking Procedure
def train_benchmark():
    epochs = 60
    batch_size = 64
    
    # Initialize both models with same architecture
    std_model = PipelineLSTM()
    pinn_model = PipelineLSTM()
    
    # Copy initial weights so both start at identical state
    pinn_model.load_state_dict(std_model.state_dict())
    
    opt_std = torch.optim.Adam(std_model.parameters(), lr=0.002)
    opt_pinn = torch.optim.Adam(pinn_model.parameters(), lr=0.002)
    
    print("🚀 Training Models (Standard LSTM vs PINN-LSTM)...")
    
    for epoch in range(epochs):
        std_model.train()
        pinn_model.train()
        
        # Shuffle indices each epoch
        permutation = torch.randperm(X_train.size()[0])
        
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            X_b, y_b = X_train[indices], y_train[indices]
            
            # --- Train Standard LSTM (MSE Only) ---
            opt_std.zero_grad()
            l_std = nn.MSELoss()(std_model(X_b), y_b)
            l_std.backward()
            opt_std.step()
            
            # --- Train PINN-LSTM (Physics Informed) ---
            opt_pinn.zero_grad()
            l_pinn = pinn_loss_function(pinn_model(X_b), y_b, X_b, epoch, epochs)
            l_pinn.backward()
            opt_pinn.step()

    # 5. Model Evaluation on Physical Scale (mm/year)
    std_model.eval()
    pinn_model.eval()
    
    with torch.no_grad():
        preds_std_scaled = std_model(X_test).numpy()
        preds_pinn_scaled = pinn_model(X_test).numpy()
        y_test_scaled = y_test.numpy()
        
        # Inverse transform back to true mm/year scale
        preds_std = scaler_y.inverse_transform(preds_std_scaled)
        preds_pinn = scaler_y.inverse_transform(preds_pinn_scaled)
        y_true = scaler_y.inverse_transform(y_test_scaled)
        
        # Calculate Real Physical MSE (mm/year)^2
        mse_std = float(np.mean((preds_std - y_true) ** 2))
        mse_pinn = float(np.mean((preds_pinn - y_true) ** 2))
        
        error_reduction = ((mse_std - mse_pinn) / mse_std) * 100.0
        
    print("\n📊 COMPARATIVE VALIDATION RESULTS (Physical Scale: mm/yr):")
    print("----------------------------------------------------------")
    print(f"Standard LSTM Test MSE:  {mse_std:.6f} mm/yr")
    print(f"PINN-LSTM Test MSE:      {mse_pinn:.6f} mm/yr")
    print(f"Error Reduction:         {error_reduction:.2f}%")
    
    # Save best PINN model weights for Streamlit Dashboard
    torch.save(pinn_model.state_dict(), "model/pinn_lstm_model.pth")
    print("✅ Model weights saved to model/pinn_lstm_model.pth")

if __name__ == "__main__":
    train_benchmark()