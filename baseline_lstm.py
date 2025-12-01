# baseline_lstm_rolling.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.stats import norm


# --------------------------
# LSTM Forecaster
# --------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden=64, layers=2, H=12):
        super().__init__()
        self.H = H
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, H)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        y = self.fc(last)
        return y.unsqueeze(-1)  # (B,H,1)


# --------------------------
# Train LSTM (ONE TIME)
# --------------------------
def train_lstm_once(X, Y, target_index, H, hidden=64, layers=2, device="cuda", epochs=50):
    device = torch.device(device)

    X_t = torch.tensor(X[:,:,target_index:target_index+1], dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y[:,:,target_index:target_index+1], dtype=torch.float32, device=device)

    dataset = torch.utils.data.TensorDataset(X_t, Y_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LSTMForecaster(1, hidden, layers, H).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()

    model.eval()
    return model


# --------------------------
# Rolling-forward Forecast
# --------------------------
def rolling_forward_lstm(
    model, X, Y, target_index=0, H=12
):
    device = next(model.parameters()).device

    N = X.shape[0]
    preds = []
    trues = []

    for i in range(N):
        x = torch.tensor(X[i:i+1, :, target_index:target_index+1], dtype=torch.float32, device=device)  # (1,L,1)
        pred = model(x).cpu().numpy()    # (1,H,1)
        preds.append(pred[0,:,0])
        trues.append(Y[i,:,target_index])

    preds = np.array(preds)   # (N,H)
    trues = np.array(trues)   # (N,H)

    # RMSE
    diff = preds - trues
    rmse = float(np.sqrt(np.mean(diff**2)))

    # sigma = residual std
    sigma = np.std(diff) + 1e-6

    # NLL
    def nll(y, mu):
        return 0.5*np.log(2*np.pi*sigma**2) + 0.5*((y-mu)**2)/(sigma**2)
    nll_mean = float(np.mean(nll(trues, preds)))

    # CRPS
    def crps(y, mu):
        z = (y-mu)/sigma
        return sigma*(z*(2*norm.cdf(z)-1) + 2*norm.pdf(z) - 1/np.sqrt(np.pi))
    crps_all = crps(trues, preds)
    crps_mean = float(np.mean(crps_all))
    crps_per_h = np.mean(crps_all, axis=0)

    return {
        "preds": preds,
        "trues": trues,
        "RMSE": rmse,
        "NLL_mean": nll_mean,
        "CRPS_mean": crps_mean,
        "CRPS_per_h": crps_per_h,
    }
