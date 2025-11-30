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
def create_dataset(df_scaled: pd.DataFrame, L: int, H: int, cond_cols):
    X, Y, C = [], [], []
    total_len = len(df_scaled)

    # +1 포함이 보통 정석 (마지막 가능한 window 포함)
    for start in range(total_len - L - H + 1):
        end_x = start + L
        end_y = end_x + H

        x = df_scaled.iloc[start:end_x].to_numpy(dtype=np.float32)  # (L,D)
        y = df_scaled.iloc[end_x:end_y].to_numpy(dtype=np.float32)  # (H,D)
        c = df_scaled.iloc[end_x - 1][cond_cols].to_numpy(dtype=np.float32)  # (cond_dim,)

        X.append(x)
        Y.append(y)
        C.append(c)

    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32), np.asarray(C, dtype=np.float32)


def _read_csv_with_date_index(csv_path: str, encoding: str, date_col: str = "Date") -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding=encoding)
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' column not found in {csv_path}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    return df


def _coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    # 콤마(1,388.90) / 공백 / 문자열 등 안전 처리
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(
            out[col].astype(str).str.replace(",", "").str.strip(),
            errors="coerce",
        )
    return out


def preprocess(
    csv_path: str,
    macro_csv_path: str,
    condition_raw_cols,
    macro_cols,
    L: int,
    H: int,
    *,
    semi_encoding: str = "latin1",
    macro_encoding: str = "utf-8-sig",
    drop_semi_cols=("PMI",),        # 반도체쪽 PMI 제거 기본값 유지
    capex_col: str = "CAPEX",
    fill_strategy: str = "ffill_bfill",  # "zero" or "ffill_bfill"
):
    """
    csv_path: 반도체 데이터 csv
    macro_csv_path: 매크로 데이터 csv
    condition_raw_cols: TimeVAE condition용 컬럼 리스트
    macro_cols: macro encoder가 보는 macro 컬럼 리스트
    """

    # 1) Load
    df_semi = _read_csv_with_date_index(csv_path, encoding=semi_encoding)
    df_macro = _read_csv_with_date_index(macro_csv_path, encoding=macro_encoding)

    # 2) Select macro cols (없으면 에러 내서 빨리 잡기)
    missing_macro = [c for c in macro_cols if c not in df_macro.columns]
    if missing_macro:
        raise ValueError(f"macro_csv is missing columns: {missing_macro}")
    df_macro = df_macro[macro_cols].copy()

    # 3) Optional: drop semi cols
    for col in drop_semi_cols:
        if col in df_semi.columns:
            df_semi = df_semi.drop(columns=[col])

    # 4) Merge by index (Date)
    df_merged = df_semi.join(df_macro, how="inner")
    df_raw = df_merged.copy()  # raw 저장(시각화/리포팅용)

    # 5) Numeric conversion (콤마/문자열 포함 안전 처리)
    df_merged = _coerce_numeric_df(df_merged)

    # 6) CAPEX log1p (0/음수 방어)
    if capex_col in df_merged.columns:
        # 음수 있으면 log1p가 NaN 되니까, 최소 0으로 클립(원하면 다른 방식 가능)
        cap = df_merged[capex_col].to_numpy()
        cap = np.clip(cap, a_min=0.0, a_max=None)
        df_merged[capex_col] = np.log1p(cap)

    # 7) Missing handling
    # condition/macro는 0으로 고정하는 기존 정책 유지 + 전체 fill 전략
    for col in condition_raw_cols:
        if col not in df_merged.columns:
            raise ValueError(f"condition column not found after merge: {col}")
    df_merged[condition_raw_cols] = df_merged[condition_raw_cols].fillna(0.0)

    df_merged[macro_cols] = df_merged[macro_cols].fillna(0.0)

    if fill_strategy == "ffill_bfill":
        df_merged = df_merged.ffill().bfill()
        df_merged = df_merged.fillna(0.0)  # 끝까지 남는 NaN 방어
    elif fill_strategy == "zero":
        df_merged = df_merged.fillna(0.0)
    else:
        raise ValueError("fill_strategy must be 'ffill_bfill' or 'zero'")

    # 8) Scaling (DataFrame 유지: feature name 경고 방지 + 순서 안전)
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_merged),
        index=df_merged.index,
        columns=df_merged.columns,
    )

    # 9) macro feature indices
    macro_feature_indices = [df_scaled.columns.get_loc(col) for col in macro_cols]

    # 10) Sliding windows
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
        # rolling_forward_test() Forecast 부분
        with torch.no_grad():
            x = torch.tensor(X_test).float().to(device)
            c = torch.tensor(C_test).float().to(device)

            macro_x = x[:, :, macro_feature_indices].permute(0, 2, 1)  # <-- 여기!

            pred, z, prior_stats = model(
                x, c, macro_x,
                y=None,
                use_prior_sampling_if_no_y=True
            )
            y_pred = pred.cpu().numpy()

        mse = np.mean((y_pred - Y_test)**2)
        mse_list.append(mse)

    return np.mean(mse_list)
