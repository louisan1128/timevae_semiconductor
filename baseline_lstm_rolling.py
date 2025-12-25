# baseline_lstm_rolling.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import norm


########################################
# Dataset
########################################
class LSTMDataset(Dataset):
    def __init__(self, X, Y, target_index=0):
        self.X = X[:, :, target_index:target_index+1]
        self.Y = Y[:, :, target_index:target_index+1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


########################################
# Model
########################################
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, layers=2, H=12):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, H)

    def forward(self, x):
        out, _ = self.lstm(x)         # (B, L, H)
        last = out[:, -1, :]          # (B, hidden)
        y = self.fc(last)             # (B, H)
        return y.unsqueeze(-1)        # (B, H, 1)


########################################
# 1) train_lstm_once (main.py에서 요구)
########################################
def train_lstm_once(
    X, Y,
    target_index=0,
    H=12,
    hidden=64,
    layers=2,
    device="cuda",
    epochs=50,
    batch_size=32
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    ds = LSTMDataset(X, Y, target_index)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = LSTMForecaster(1, hidden, layers, H).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for ep in range(1, epochs+1):
        total = 0
        for xb, yb in dl:
            xb = xb.float().to(device)
            yb = yb.float().to(device)

            optim.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optim.step()

            total += loss.item()

        if ep % 10 == 0 or ep == 1:
            print(f"[LSTM train ep {ep}] loss={total/len(dl):.4f}")

    return model


########################################
# 2) rolling_forward_lstm (probabilistic)
########################################
def rolling_forward_lstm(
    model,
    X, Y,
    target_index=0,
    H=12,
    num_samples=500,
    lower_q=10,
    upper_q=90
):
    device = next(model.parameters()).device

    N = X.shape[0]
    preds = []
    trues = []

    # Residual 저장 → sigma 추정
    all_resid = []

    # 1차 패스: deterministic 예측
    for i in range(N):
        x_i = torch.tensor(X[i:i+1, :, target_index:target_index+1],
                           dtype=torch.float32, device=device)
        y_i = Y[i, :, target_index]      # (H,)

        with torch.no_grad():
            pred = model(x_i).cpu().numpy()[0, :, 0]  # (H,)

        preds.append(pred)
        trues.append(y_i)
        all_resid.extend(list(pred - y_i))

    preds = np.array(preds)
    trues = np.array(trues)

    # sigma (Gaussian assumption)
    sigma = np.std(all_resid) + 1e-6

    # ============================
    # Probabilistic metrics
    # ============================
    nll_list = []
    crps_list = []
    crps_h_list = []
    cov_list = []
    sharp_list = []

    from scipy.stats import norm

    for i in range(N):
        p = preds[i]  # (H,)
        t = trues[i]  # (H,)

        # ---------- NLL ----------
        nll = 0.5*np.log(2*np.pi*(sigma**2)) + 0.5*((t - p)**2)/(sigma**2)
        nll_list.append(nll.mean())

        # ---------- CRPS ----------
        z = (t - p) / sigma
        crps_h = sigma * (z*(2*norm.cdf(z)-1) + 2*norm.pdf(z) - 1/np.sqrt(np.pi))
        crps_list.append(crps_h.mean())
        crps_h_list.append(crps_h)

        # ---------- Coverage & Sharpness ----------
        zL = norm.ppf(lower_q/100)
        zU = norm.ppf(upper_q/100)

        lower = p + sigma*zL
        upper = p + sigma*zU

        inside = (t >= lower) & (t <= upper)
        cov_list.append(inside.mean())
        sharp_list.append((upper - lower).mean())

    # ---------- RMSE ----------
    rmse = np.sqrt(np.mean((preds - trues)**2))

    return {
        "preds": preds,
        "trues": trues,
        "RMSE": float(rmse),
        "NLL_mean": float(np.mean(nll_list)),
        "CRPS_mean": float(np.mean(crps_list)),
        "CRPS_per_h": np.mean(crps_h_list, axis=0),
        "Coverage_80%": float(np.mean(cov_list)),
        "Sharpness_80%": float(np.mean(sharp_list)),
    }
