# baseline_lstm.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# Dataset (target 1차원만 사용)
# -------------------------------
class LSTMDataset(Dataset):
    def __init__(self, X, Y, target_index=0):
        """
        X: (N, L, D)
        Y: (N, H, D)
        """
        self.X = X[:, :, target_index:target_index+1]  # (N, L, 1)
        self.Y = Y[:, :, target_index:target_index+1]  # (N, H, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# -------------------------------
# LSTM Forecaster
# -------------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, H=12):
        super().__init__()
        self.H = H
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, H)  # output: (B, H)

    def forward(self, x):
        """
        x: (B, L, 1)
        return: (B, H, 1)
        """
        out, _ = self.lstm(x)       # (B, L, H)
        last = out[:, -1, :]        # (B, H)
        y = self.fc(last)           # (B, H)
        return y.unsqueeze(-1)      # (B, H, 1)


# -------------------------------
# Train + Evaluate
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

    dataset = LSTMDataset(X, Y, target_index=target_index)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMForecaster(
        input_dim=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        H=H
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for ep in range(1, epochs+1):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device).float()
            y_batch = y_batch.to(device).float()

            optim.zero_grad()
            y_hat = model(x_batch)
            loss = criterion(y_hat, y_batch)
            loss.backward()
            optim.step()

            total_loss += loss.item() * x_batch.size(0)

        if ep % 10 == 0 or ep == 1:
            print(f"[LSTM ep {ep:03d}] train MSE = {total_loss / len(dataset):.4f}")

    # -------- eval (전체 X에 대한 예측) ----------
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(dataset.X, dtype=torch.float32, device=device)
        y_pred = model(X_t).cpu().numpy()     # (N, H, 1)
        y_true = dataset.Y                   # (N, H, 1)

    # point metrics (target 1차원만)
    diff = y_pred - y_true
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))

    return {
        "preds": y_pred,   # (N, H, 1)
        "trues": y_true,   # (N, H, 1)
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
    }
