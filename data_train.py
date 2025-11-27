# ===========================================
# data_train_utils.py
# (Dataset + Preprocessing + Training + Backtests)
# ===========================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

from model import Encoder, Decoder, ConditionalPrior, TimeVAE



# -------------------------------
# Dataset
# -------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, c):
        self.x = x
        self.y = y
        self.c = c

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.c[idx]

# -------------------------------
# Create Dataset (Sliding Window)
# -------------------------------
def create_dataset(df_scaled, L, H, cond_cols):
    X, Y, C = [], [], []
    total_len = len(df_scaled)

    for start in range(total_len - L - H):
        end_x = start + L
        end_y = end_x + H

        x = df_scaled.iloc[start:end_x].values      # (L,D)
        y = df_scaled.iloc[end_x:end_y].values      # (H,D)
        c = df_scaled.iloc[end_x - 1][cond_cols].values  # (cond_dim,)

        X.append(x)
        Y.append(y)
        C.append(c)

    return (
        np.array(X, dtype=np.float32),
        np.array(Y, dtype=np.float32),
        np.array(C, dtype=np.float32),
    )


# -------------------------------
# Preprocess (Load CSV → Scaling)
# -------------------------------
def preprocess(csv_path, condition_raw_cols, L, H):
    df = pd.read_csv(csv_path, encoding="latin1")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    df_raw = df.copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["CAPEX"] = np.log1p(df["CAPEX"])
    df[condition_raw_cols] = df[condition_raw_cols].fillna(0)

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns
    )

    X, Y, C = create_dataset(df_scaled, L, H, condition_raw_cols)

    print("X shape:", X.shape)  # (N,L,D)
    print("Y shape:", Y.shape)  # (N,H,D)
    print("C shape:", C.shape)  # (N,cond_dim)

    return X, Y, C, scaler, df_raw, df_scaled



# -------------------------------
# Training
# -------------------------------
def train_model(
    X_train, Y_train, C_train,
    latent_dim, cond_dim, hidden,
    H_len, beta, lr, epochs, batch_size, device
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dim = X_train.shape[-1]

    train_dataset = TimeSeriesDataset(X_train, Y_train, C_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H_len).to(device)
    prior = ConditionalPrior(cond_dim, latent_dim, hidden).to(device)

    model = TimeVAE(encoder, decoder, prior, latent_dim, beta).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("======== Training start ========")
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        num_batches = 0

        for x, y, c in train_loader:
            x, y, c = x.to(device), y.to(device), c.to(device)

            loss, recon, kl, _, _ = model(x, c, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()
            num_batches += 1

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"Loss: {epoch_loss/num_batches:.4f} | "
            f"Recon: {epoch_recon/num_batches:.4f} | "
            f"KL: {epoch_kl/num_batches:.4f}"
        )

    torch.save(model.state_dict(), "timevae_ctvae_prior.pth")
    print("Saved model → timevae_ctvae_prior.pth")
    return model


# ======================================================================
# 1) Rolling Backtest (고정 모델로 1-step Forecast)
# ======================================================================

def rolling_backtest(model_path, X, Y, C,
                     latent_dim, cond_dim, hidden, H, beta,
                     device="cuda"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dim = X.shape[-1]

    # 모델 구성 & 가중치 로드
    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H).to(device)
    prior = ConditionalPrior(cond_dim, latent_dim, hidden).to(device)

    model = TimeVAE(encoder, decoder, prior, latent_dim, beta).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mses = []

    # 모든 시점에서 고정 모델로 예측
    for t in range(len(X)):
        x = torch.tensor(X[t:t+1]).float().to(device)
        c = torch.tensor(C[t:t+1]).float().to(device)
        y_true = Y[t:t+1]

        with torch.no_grad():
            y_pred, _, _ = model(x, c, y=None, use_prior_sampling_if_no_y=True)
            y_pred = y_pred.cpu().numpy()

        mses.append(np.mean((y_pred - y_true)**2))

    return np.mean(mses)


# ======================================================================
# 2) Rolling Forward Test (Expanding Window, 매 앵커마다 재학습)
# ======================================================================

def rolling_forward_test(X, Y, C,
                         latent_dim, cond_dim, hidden, H,
                         beta, lr, epochs, batch_size,
                         device="cuda", L_window=None):
    """
    Expanding Window Rolling-Forward Backtest (정석)
    매 anchor t에 대해:
        - Train: 0 ~ t
        - Test: t -> t+H 예측
        - 모델 매번 재학습
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = len(X)

    mse_list = []

    for anchor in range(1, N - 1):
        print(f"[Rolling-Forward] Anchor {anchor}/{N}")

        # 1) Expanding Window train set
        X_train = X[:anchor]
        Y_train = Y[:anchor]
        C_train = C[:anchor]

        # 2) Test one-step (anchor)
        X_test = X[anchor:anchor + 1]
        Y_test = Y[anchor:anchor + 1]
        C_test = C[anchor:anchor + 1]

        # 3) 매번 모델 재학습
        model = train_model(
            X_train, Y_train, C_train,
            latent_dim, cond_dim, hidden,
            H, beta, lr, epochs, batch_size, device
        )

        # 4) Forecast
        with torch.no_grad():
            x = torch.tensor(X_test).float().to(device)
            c = torch.tensor(C_test).float().to(device)

            y_pred, _, _ = model(x, c, y=None, use_prior_sampling_if_no_y=True)
            y_pred = y_pred.cpu().numpy()

        mse = np.mean((y_pred - Y_test)**2)
        mse_list.append(mse)

    return np.mean(mse_list)