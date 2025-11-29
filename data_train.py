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
from macro_pretrain import MacroEncoder


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
def preprocess(csv_path, macro_csv_path, condition_raw_cols, macro_cols, L, H):
    """
    semi_csv_path : 반도체 데이터 csv (data.csv)
    macro_csv_path: 매크로 데이터 csv (macro.csv)
    condition_raw_cols : TimeVAE condition용 컬럼 이름 리스트 (5개)
    macro_cols : macro encoder가 보는 macro 컬럼 이름 리스트 (6개)
    """
    
    df_semi = pd.read_csv(csv_path, encoding="latin1")
    df_semi["Date"] = pd.to_datetime(df_semi["Date"])
    df_semi = df_semi.set_index("Date")


    # 2) 매크로 데이터
    df_macro = pd.read_csv("macro.csv", encoding="utf-8-sig")
    df_macro["Date"] = pd.to_datetime(df_macro["Date"])
    df_macro = df_macro.set_index("Date")

    # 매크로에서 필요한 컬럼만 남기기
    df_macro = df_macro[macro_cols]

    # 3) 날짜 기준으로 inner join (공통 구간만 사용)
    df_semi = df_semi.drop(columns=["PMI"])    # 반도체 PMI 제거
    df_merged = pd.merge(df_semi, df_macro, on="Date", how="inner")

    df_raw = df_merged.copy()

    # 4) numeric 변환
    for col in df_merged.columns:
        df_merged[col] = pd.to_numeric(df_merged[col], errors="coerce")

    # 5) CAPEX 로그 변환
    if "CAPEX" in df_merged.columns:
        df_merged["CAPEX"] = np.log1p(df_merged["CAPEX"])


    # 6) 결측치 처리
    #    - condition 컬럼은 0으로
    df_merged[condition_raw_cols] = df_merged[condition_raw_cols].fillna(0)
    #    - macro 컬럼도 0으로
    df_merged[macro_cols] = df_merged[macro_cols].fillna(0)
    #    - 그 외는 그냥 ffll/bfill 또는 0 등 필요하면 추가 처리 가능
    df_merged = df_merged.fillna(0)


     # 7) 스케일링
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_merged),
        index=df_merged.index,
        columns=df_merged.columns
    )

    # 8) macro feature index (X에서 macro만 뽑기 위해)
    macro_feature_indices = [
        df_scaled.columns.get_loc(col) for col in macro_cols
    ]
    
    
    # 9) Sliding window 만들기
    X, Y, C = create_dataset(df_scaled, L, H, condition_raw_cols)

    print("X shape:", X.shape)  # (N,L,D)
    print("Y shape:", Y.shape)  # (N,H,D)
    print("C shape:", C.shape)  # (N,cond_dim)
    print("macro_feature_indices:", macro_feature_indices)
    

    return X, Y, C, scaler, df_raw, df_scaled, macro_feature_indices



# -------------------------------
# Training
# -------------------------------
def train_model(
     X_train, Y_train, C_train,
    latent_dim, cond_dim, hidden,
    H_len, beta, lr, epochs, batch_size, device,
    macro_feature_indices,
    macro_latent_dim=32,
    macro_hidden_dim=128,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dim = X_train.shape[-1]

    train_dataset = TimeSeriesDataset(X_train, Y_train, C_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 1) Pretrained MacroEncoder 로드 (input_dim = len(macro_feature_indices))
    macro_input_dim = len(macro_feature_indices)
    macro_encoder = MacroEncoder(
        input_dim=macro_input_dim,
        hidden_dim=macro_hidden_dim,
        latent_dim=macro_latent_dim
    ).to(device)
    macro_encoder.load_state_dict(torch.load("macro_encoder.pth", map_location=device))
    macro_encoder.eval()
    for p in macro_encoder.parameters():
        p.requires_grad = False

    
    # 2) TimeVAE 구성
    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H_len).to(device)
    prior = ConditionalPrior(
        cond_dim,
        macro_latent_dim=macro_latent_dim,
        latent_dim=latent_dim,
        hidden=hidden
    ).to(device)


    model = TimeVAE(
        encoder, decoder, prior,
        macro_encoder=macro_encoder,
        latent_dim=latent_dim,
        beta=beta
    ).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("======== Training start ========")
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        num_batches = 0

        for x, y, c in train_loader:
            x, y, c = x.to(device), y.to(device), c.to(device)

            # macro_x: (B, macro_dim, L)
            macro_x = x[:, :, macro_feature_indices]      # (B, L, macro_dim)
            macro_x = macro_x.permute(0, 2, 1)           # (B, macro_dim, L)

            loss, recon, kl, _, _, _ = model(
                x, c, macro_x, y,
                use_prior_sampling_if_no_y=False
            )

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

def rolling_backtest( model_path, X, Y, C,
    latent_dim, cond_dim, hidden, H, beta,
    macro_feature_indices,
    macro_latent_dim=32,
    macro_hidden_dim=128,
    device="cuda"
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dim = X.shape[-1]

    # Pretrained MacroEncoder 로드
    macro_input_dim = len(macro_feature_indices)
    macro_encoder = MacroEncoder(
        input_dim=macro_input_dim,
        hidden_dim=macro_hidden_dim,
        latent_dim=macro_latent_dim
    ).to(device)
    macro_encoder.load_state_dict(torch.load("macro_encoder.pth", map_location=device))
    macro_encoder.eval()
    for p in macro_encoder.parameters():
        p.requires_grad = False

    # 모델 구성 & 가중치 로드
    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H).to(device)
    prior = ConditionalPrior(
        cond_dim, macro_latent_dim,
        latent_dim, hidden
    ).to(device)

    model = TimeVAE(encoder, decoder, prior, macro_encoder, latent_dim, beta).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mses = []

    # 모든 시점에서 고정 모델로 예측
    for t in range(len(X)):
        x = torch.tensor(X[t:t+1]).float().to(device)
        c = torch.tensor(C[t:t+1]).float().to(device)
        macro_x = x[:, :, :6].permute(0, 2, 1)
        y_true = Y[t:t+1]

        # macro_x: (1, macro_dim, L)
        macro_x = x[:, :, macro_feature_indices].permute(0, 2, 1)

        with torch.no_grad():
            pred, z, prior_stats = model(
                x, c, macro_x,
                y=None,
                use_prior_sampling_if_no_y=True
            )
            y_pred = pred.cpu().numpy()

        mses.append(np.mean((y_pred - y_true)**2))

    return np.mean(mses)


# ======================================================================
# 2) Rolling Forward Test (Expanding Window, 매 앵커마다 재학습)
# ======================================================================

def rolling_forward_test(X, Y, C,
    latent_dim, cond_dim, hidden, H,
    beta, lr, epochs, batch_size,
    macro_feature_indices,
    macro_latent_dim=32,
    macro_hidden_dim=128,
    device="cuda",
    L_window=None):
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
            H, beta, lr, epochs, batch_size, device,
            macro_feature_indices=macro_feature_indices,
            macro_latent_dim=macro_latent_dim,
            macro_hidden_dim=macro_hidden_dim,
        )

        # 4) Forecast
        with torch.no_grad():
            x = torch.tensor(X_test).float().to(device)
            c = torch.tensor(C_test).float().to(device)
            macro_x = x[:, :, :6].permute(0, 2, 1)

            pred, z, prior_stats = model(
                x, c, macro_x,
                y=None,
                use_prior_sampling_if_no_y=True
            )
            y_pred = pred.cpu().numpy()

        mse = np.mean((y_pred - Y_test)**2)
        mse_list.append(mse)

    return np.mean(mse_list)