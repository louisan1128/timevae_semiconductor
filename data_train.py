# ===========================================
# data_train_utils.py
# (Dataset + Preprocessing + Training + Backtests)
# ===========================================

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from model import Encoder, Decoder, ConditionalPrior, TimeVAE
from macro_pretrain import MacroEncoder


# -------------------------------
# Dataset
# -------------------------------
class TimeSeriesDataset(Dataset):
    """
    Returns:
      x      : (L, D_all)
      y      : (H, D_all)
      c      : (cond_dim,)
      macro_x: (macro_dim, L)
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, c: np.ndarray, macro_x: np.ndarray):
        self.x = x
        self.y = y
        self.c = c
        self.macro_x = macro_x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.c[idx], self.macro_x[idx]


# -------------------------------
# Helpers
# -------------------------------
def _read_csv_with_date_index(csv_path: str, encoding: str, date_col: str = "Date") -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding=encoding)
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' column not found in {csv_path}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    return df


def _coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(
            out[col].astype(str).str.replace(",", "").str.strip(),
            errors="coerce",
        )
    return out


def _make_macro_features_pretrain_style(df_macro_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Pretrain에서 썼던 것과 동일하게 생성:
      PMI, GS10, LOG_M2=log(M2SL), UNRATE, INFLATION=diff(log(CPIAUCSL)), LOG_INDPRO=log(INDPRO)
    """
    required = ["PMI", "GS10", "M2SL", "UNRATE", "CPIAUCSL", "INDPRO"]
    missing = [c for c in required if c not in df_macro_raw.columns]
    if missing:
        raise ValueError(f"macro_csv is missing columns: {missing}")

    # 결측/보간: pretrain과 유사하게 forward-only로 최대한 맞추고, 남는 건 ffill
    dm = df_macro_raw[required].copy()
    dm = _coerce_numeric_df(dm)
    dm = dm.ffill().interpolate(limit_direction="forward").ffill()

    # log 변환 안정성(0/음수 방지)
    m2 = np.clip(dm["M2SL"].to_numpy(dtype=float), a_min=1e-12, a_max=None)
    cpi = np.clip(dm["CPIAUCSL"].to_numpy(dtype=float), a_min=1e-12, a_max=None)
    ip = np.clip(dm["INDPRO"].to_numpy(dtype=float), a_min=1e-12, a_max=None)

    feat = pd.DataFrame(
        {
            "PMI": dm["PMI"].to_numpy(dtype=float),
            "GS10": dm["GS10"].to_numpy(dtype=float),
            "LOG_M2": np.log(m2),
            "UNRATE": dm["UNRATE"].to_numpy(dtype=float),
            "INFLATION": np.log(cpi),
            "LOG_INDPRO": np.log(ip),
        },
        index=dm.index,
    )

    # INFLATION은 diff(log CPI)
    feat["INFLATION"] = feat["INFLATION"].diff()

    # diff로 생긴 NaN 제거
    feat = feat.dropna()
    return feat


# -------------------------------
# Create Dataset (Sliding Window)
# -------------------------------
def create_dataset_with_macro(
    df_all_scaled: pd.DataFrame,
    df_macro_scaled: pd.DataFrame,
    L: int,
    H: int,
    cond_cols,
):
    """
    df_all_scaled  : (T, D_all)  - CT-VAE의 x/y/c를 생성할 전체 입력 (scaled)
    df_macro_scaled: (T, D_macro)- MacroEncoder 입력용 (scaled by macro_scaler.pkl), pretrain-style features
    """
    if not df_all_scaled.index.equals(df_macro_scaled.index):
        # 공통 인덱스로 재정렬 (inner)
        common = df_all_scaled.index.intersection(df_macro_scaled.index)
        df_all_scaled = df_all_scaled.loc[common]
        df_macro_scaled = df_macro_scaled.loc[common]

    for col in cond_cols:
        if col not in df_all_scaled.columns:
            raise ValueError(f"condition column not found in df_all_scaled: {col}")

    X, Y, C, MX = [], [], [], []
    total_len = len(df_all_scaled)

    for start in range(total_len - L - H + 1):
        end_x = start + L
        end_y = end_x + H

        x = df_all_scaled.iloc[start:end_x].to_numpy(dtype=np.float32)   # (L, D_all)
        y = df_all_scaled.iloc[end_x:end_y].to_numpy(dtype=np.float32)   # (H, D_all)
        c = df_all_scaled.iloc[end_x - 1][cond_cols].to_numpy(dtype=np.float32)  # (cond_dim,)

        macro_seq = df_macro_scaled.iloc[start:end_x].to_numpy(dtype=np.float32)  # (L, D_macro)
        macro_seq = np.ascontiguousarray(macro_seq.T)  # (D_macro, L)

        X.append(x)
        Y.append(y)
        C.append(c)
        MX.append(macro_seq)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(Y, dtype=np.float32),
        np.asarray(C, dtype=np.float32),
        np.asarray(MX, dtype=np.float32),  # (N, D_macro, L)
    )


# -------------------------------
# Preprocess (Load CSV → Align → Feature Eng → Scaling)
# -------------------------------
def preprocess(
    csv_path: str,
    macro_csv_path: str,
    condition_raw_cols,
    macro_cols,   # (raw macro cols) kept for compatibility; macro feature eng uses required columns
    L: int,
    H: int,
    *,
    semi_encoding: str = "latin1",
    macro_encoding: str = "utf-8-sig",
    drop_semi_cols=("PMI",),        # 반도체쪽 PMI 제거 기본값 유지
    capex_col: str = "CAPEX",
    fill_strategy: str = "ffill_bfill",  # "zero" or "ffill_bfill"
    #
    macro_scaler_path: str = "macro_scaler.pkl",
    use_macro_scaler_file: bool = True,
):
    """
    Returns:
      X, Y, C, MACRO_X,
      scaler_all, macro_scaler,
      df_raw_all, df_all_scaled,
      macro_feature_indices (for convenience; in df_all_scaled),
      macro_feat_cols (actual macro feature names used)
    """

    # 1) Load
    df_semi = _read_csv_with_date_index(csv_path, encoding=semi_encoding)
    df_macro = _read_csv_with_date_index(macro_csv_path, encoding=macro_encoding)

    # 2) drop semi cols (e.g., PMI duplicates)
    for col in drop_semi_cols:
        if col in df_semi.columns:
            df_semi = df_semi.drop(columns=[col])

    # 3) Macro feature engineering (pretrain-style)
    df_macro_feat = _make_macro_features_pretrain_style(df_macro)

    macro_feat_cols = ["PMI", "GS10", "LOG_M2", "UNRATE", "INFLATION", "LOG_INDPRO"]
    df_macro_feat = df_macro_feat[macro_feat_cols].copy()

    # 4) Merge: use engineered macro features in the main input too (recommended for consistency)
    df_all = df_semi.join(df_macro_feat, how="inner")

    # raw copy (전처리 전 상태)
    df_raw_all = df_all.copy()

    # 5) Numeric conversion (safe)
    df_all = _coerce_numeric_df(df_all)

    # 6) CAPEX log1p (0/음수 방어)
    if capex_col in df_all.columns:
        cap = df_all[capex_col].to_numpy(dtype=float)
        cap = np.clip(cap, a_min=0.0, a_max=None)
        df_all[capex_col] = np.log1p(cap)

    # 7) Missing handling
    # condition cols must exist
    for col in condition_raw_cols:
        if col not in df_all.columns:
            raise ValueError(f"condition column not found after merge: {col}")

    if fill_strategy == "ffill_bfill":
        df_all = df_all.ffill().bfill().fillna(0.0)
    elif fill_strategy == "zero":
        df_all = df_all.fillna(0.0)
    else:
        raise ValueError("fill_strategy must be 'ffill_bfill' or 'zero'")

    # 8) Scaling for CT-VAE x/y/c (전체 피처)
    scaler_all = StandardScaler()
    df_all_scaled = pd.DataFrame(
        scaler_all.fit_transform(df_all),
        index=df_all.index,
        columns=df_all.columns,
    )

    # 9) Scaling for macro encoder input (macro_scaler.pkl, pretrain distribution)
    if use_macro_scaler_file:
        macro_scaler = joblib.load(macro_scaler_path)
    else:
        macro_scaler = StandardScaler().fit(df_macro_feat.loc[df_all.index].values)

    df_macro_scaled = pd.DataFrame(
        macro_scaler.transform(df_macro_feat.loc[df_all.index].values),
        index=df_all.index,
        columns=macro_feat_cols,
    )

    # 10) Convenience indices: where macro features live in df_all_scaled
    # (이제 macro_x는 별도 df_macro_scaled로 쓰지만, 그래도 디버깅/시각화용으로 유용)
    macro_feature_indices = [df_all_scaled.columns.get_loc(col) for col in macro_feat_cols]

    # 11) Sliding windows (X/Y/C + MACRO_X)
    X, Y, C, MACRO_X = create_dataset_with_macro(
        df_all_scaled, df_macro_scaled,
        L=L, H=H,
        cond_cols=condition_raw_cols,
    )

    print("X shape:", X.shape)            # (N, L, D_all)
    print("Y shape:", Y.shape)            # (N, H, D_all)
    print("C shape:", C.shape)            # (N, cond_dim)
    print("MACRO_X shape:", MACRO_X.shape)  # (N, D_macro=6, L)
    print("macro_feat_cols:", macro_feat_cols)
    print("macro_feature_indices (in df_all_scaled):", macro_feature_indices)

    return (
        X, Y, C, MACRO_X,
        scaler_all, macro_scaler,
        df_raw_all, df_all_scaled,
        macro_feature_indices, macro_feat_cols
    )


# -------------------------------
# Training
# -------------------------------
def train_model(
    X_train,
    Y_train,
    C_train,
    MACRO_X_train,
    latent_dim,
    cond_dim,
    hidden,
    H_len,
    beta,
    lr,
    epochs,
    batch_size,
    device,
    *,
    macro_latent_dim=32,
    macro_hidden_dim=128,
    macro_encoder_path: str = "macro_encoder.pth",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    out_dim = X_train.shape[-1]

    train_dataset = TimeSeriesDataset(X_train, Y_train, C_train, MACRO_X_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 1) Pretrained MacroEncoder 로드 (pretrain style macro_x: (B, 6, L))
    macro_input_dim = MACRO_X_train.shape[1]
    macro_encoder = MacroEncoder(
        input_dim=macro_input_dim,
        hidden_dim=macro_hidden_dim,
        latent_dim=macro_latent_dim
    ).to(device)

    macro_encoder.load_state_dict(torch.load(macro_encoder_path, map_location=device))
    macro_encoder.eval()
    for p in macro_encoder.parameters():
        p.requires_grad = False

    # 2) TimeVAE 구성
    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H_len).to(device)
    prior = ConditionalPrior(
        cond_dim=cond_dim,
        macro_latent_dim=macro_latent_dim,
        latent_dim=latent_dim,
        hidden=hidden
    ).to(device)

    model = TimeVAE(
        encoder=encoder,
        decoder=decoder,
        prior=prior,
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

        for x, y, c, macro_x in train_loader:
            x = x.to(device)
            y = y.to(device)
            c = c.to(device)
            macro_x = macro_x.to(device)  # (B, 6, L)

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
def rolling_backtest(
    model_path,
    X,
    Y,
    C,
    MACRO_X,
    latent_dim,
    cond_dim,
    hidden,
    H,
    beta,
    *,
    macro_latent_dim=32,
    macro_hidden_dim=128,
    device="cuda",
    macro_encoder_path: str = "macro_encoder.pth",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    out_dim = X.shape[-1]

    # Pretrained MacroEncoder 로드
    macro_input_dim = MACRO_X.shape[1]
    macro_encoder = MacroEncoder(
        input_dim=macro_input_dim,
        hidden_dim=macro_hidden_dim,
        latent_dim=macro_latent_dim
    ).to(device)
    macro_encoder.load_state_dict(torch.load(macro_encoder_path, map_location=device))
    macro_encoder.eval()
    for p in macro_encoder.parameters():
        p.requires_grad = False

    # 모델 구성 & 가중치 로드
    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H).to(device)
    prior = ConditionalPrior(cond_dim, macro_latent_dim, latent_dim, hidden).to(device)

    model = TimeVAE(encoder, decoder, prior, macro_encoder, latent_dim, beta).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mses = []

    for t in range(len(X)):
        x = torch.tensor(X[t:t+1]).float().to(device)
        c = torch.tensor(C[t:t+1]).float().to(device)
        macro_x = torch.tensor(MACRO_X[t:t+1]).float().to(device)  # (1,6,L)
        y_true = Y[t:t+1]

        with torch.no_grad():
            mean, dist, z, src, post_stats, prior_stats = model(
                x, c, macro_x,
                y=None,
                use_prior_sampling_if_no_y=True
            )
            y_pred = mean.cpu().numpy()

        mses.append(np.mean((y_pred - y_true) ** 2))

    return float(np.mean(mses))


# ======================================================================
# 2) Rolling Forward Test (Expanding Window, 매 앵커마다 재학습)
# ======================================================================
def rolling_forward_test(
    X,
    Y,
    C,
    MACRO_X,
    latent_dim,
    cond_dim,
    hidden,
    H,
    beta,
    lr,
    epochs,
    batch_size,
    *,
    macro_latent_dim=32,
    macro_hidden_dim=128,
    device="cuda",
    macro_encoder_path: str = "macro_encoder.pth",
):
    """
    Expanding Window Rolling-Forward Backtest (정석)
    매 anchor t에 대해:
      - Train: 0 ~ t
      - Test : t에서 1개 window 예측
      - 모델 매번 재학습
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    N = len(X)

    mse_list = []

    for anchor in range(1, N - 1):
        print(f"[Rolling-Forward] Anchor {anchor}/{N}")

        # expanding train
        X_train = X[:anchor]
        Y_train = Y[:anchor]
        C_train = C[:anchor]
        MX_train = MACRO_X[:anchor]

        # test one
        X_test = X[anchor:anchor + 1]
        Y_test = Y[anchor:anchor + 1]
        C_test = C[anchor:anchor + 1]
        MX_test = MACRO_X[anchor:anchor + 1]

        model = train_model(
            X_train, Y_train, C_train, MX_train,
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            hidden=hidden,
            H_len=H,
            beta=beta,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            device=str(device),
            macro_latent_dim=macro_latent_dim,
            macro_hidden_dim=macro_hidden_dim,
            macro_encoder_path=macro_encoder_path,
        )

        with torch.no_grad():
            x = torch.tensor(X_test).float().to(device)
            c = torch.tensor(C_test).float().to(device)
            macro_x = torch.tensor(MX_test).float().to(device)

            mean, dist, z, src, post_stats, prior_stats = model(
                x, c, macro_x,
                y=None,
                use_prior_sampling_if_no_y=True
            )
            y_pred = mean.cpu().numpy()

        mse = float(np.mean((y_pred - Y_test) ** 2))
        mse_list.append(mse)

    return float(np.mean(mse_list))
