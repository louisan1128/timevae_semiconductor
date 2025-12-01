# baseline_lstm.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import norm

# -------------------------------
# Dataset (target 1차원만 사용)
# -------------------------------
class LSTMDataset(Dataset):
    def __init__(self, X, Y, target_index=0):
        self.X = X[:, :, target_index:target_index+1]  # (N, L, 1)
        self.Y = Y[:, :, target_index:target_index+1]  # (N, H, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# -------------------------------
# LSTM Model
# -------------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, H=12):
        super().__init__()
        self.H = H
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, H)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        y = self.fc(last)
        return y.unsqueeze(-1)  # (B, H, 1)

# -------------------------------
# Probabilistic Metrics
# -------------------------------
def gaussian_nll(y, mu, sigma):
    eps = 1e-6
    return 0.5 * np.log(2*np.pi*(sigma**2) + eps) + 0.5 * ((y - mu)**2)/(sigma**2 + eps)

def gaussian_crps(y, mu, sigma):
    z = (y - mu) / sigma
    return sigma * (z * (2*norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1/np.sqrt(np.pi))

def gaussian_coverage_and_sharpness(y_true, mu, sigma, lower_q=10, upper_q=90):
    z_low = norm.ppf(lower_q / 100)
    z_up  = norm.ppf(upper_q / 100)

    lower = mu + sigma * z_low
    upper = mu + sigma * z_up

    inside = (y_true >= lower) & (y_true <= upper)
    coverage = inside.mean()

    sharpness = (upper - lower).mean()
    return coverage, sharpness

# -------------------------------
# LSTM Training + Probabilistic Evaluation
# -------------------------------
def train_lstm_baseline(
    X, Y,
    target_index=0,
    H=12,
    hidden_dim=64,
    num_layers=2,
    epochs=50,
    batch_size=32,
    lr=1e-3,
    device="cuda"
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    dataset = LSTMDataset(X, Y, target_index)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMForecaster(1, hidden_dim, num_layers, H).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # ---------------- Training ----------------
    model.train()
    for ep in range(1, epochs+1):
        total = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().to(device)

            optim.zero_grad()
            y_hat = model(x_batch)
            loss = criterion(y_hat, y_batch)
            loss.backward()
            optim.step()

            total += loss.item() * x_batch.size(0)

        if ep % 10 == 0 or ep == 1:
            print(f"[LSTM ep {ep:03d}] train MSE = {total / len(dataset):.4f}")

    # ---------------- Deterministic Prediction ----------------
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(dataset.X, dtype=torch.float32, device=device)
        y_pred = model(X_t).cpu().numpy()  # (N, H, 1)
        y_true = dataset.Y                # (N, H, 1)

    # ---------------- Point Metrics ----------------
    diff = y_pred - y_true
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))

    # ---------------- Residual-based Uncertainty ----------------
    # residuals: flatten over all N,H
    resid = (y_pred - y_true).reshape(-1)
    sigma = float(np.std(resid) + 1e-6)  # constant sigma

    # ---------------- Probabilistic Metrics ----------------
    N = y_pred.shape[0]
    nll_list = []
    crps_list = []
    cov_list = []
    sharp_list = []

    for i in range(N):
        mu = y_pred[i,:,0]   # (H,)
        y_i = y_true[i,:,0]  # (H,)

        nll_list.append(np.mean(gaussian_nll(y_i, mu, sigma)))
        crps_list.append(np.mean(gaussian_crps(y_i, mu, sigma)))

        cov_i, sharp_i = gaussian_coverage_and_sharpness(y_i, mu, sigma)
        cov_list.append(cov_i)
        sharp_list.append(sharp_i)

    return {
        "preds": y_pred,
        "trues": y_true,
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "NLL_mean": float(np.mean(nll_list)),
        "CRPS_mean": float(np.mean(crps_list)),
        "Coverage_80%": float(np.mean(cov_list)),
        "Sharpness_80%": float(np.mean(sharp_list)),
    }
