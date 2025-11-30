# ===========================================
# data_train.py
# (Dataset + Preprocessing + Training + Backtests)
# ===========================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import joblib  # for macro_scaler.pkl (optional)
except Exception:
    joblib = None

from model import Encoder, Decoder, ConditionalPrior, TimeVAE
from macro_pretrain import MacroEncoder


# -------------------------------
# Dataset
# -------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, c, macro_x):
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


def _make_macro_features(df_macro_base: pd.DataFrame, macro_cols_base) -> (pd.DataFrame, list):
    """
    Input macro_cols_base (예: ['PMI','GS10','M2SL','UNRATE','CPIAUCSL','INDPRO'])
    Output macro_feat_df with 6 cols:
      ['PMI','GS10','LOG_M2','UNRATE','INFLATION','LOG_INDPRO']
    """
    need = list(macro_cols_base)
    missing = [c for c in need if c not in df_macro_base.columns]
    if missing:
        raise ValueError(f"macro.csv missing columns: {missing}")

    base = df_macro_base[need].copy()
    base = _coerce_numeric_df(base)

    # --- features ---
    feat = pd.DataFrame(index=base.index)
    feat["PMI"] = base["PMI"]
    feat["GS10"] = base["GS10"]

    # LOG_M2: log1p(M2SL)
    feat["LOG_M2"] = np.log1p(np.clip(base["M2SL"].to_numpy(dtype=float), a_min=0.0, a_max=None))

    feat["UNRATE"] = base["UNRATE"]

    # INFLATION: 100 * diff(log(CPI))  (monthly log inflation)
    cpi = np.clip(base["CPIAUCSL"].to_numpy(dtype=float), a_min=1e-8, a_max=None)
    feat["INFLATION"] = 100.0 * np.diff(np.log(cpi), prepend=np.log(cpi[0]))

    # LOG_INDPRO: log(INDPRO)
    indpro = np.clip(base["INDPRO"].to_numpy(dtype=float), a_min=1e-8, a_max=None)
    feat["LOG_INDPRO"] = np.log(indpro)

    macro_feat_cols = ["PMI", "GS10", "LOG_M2", "UNRATE", "INFLATION", "LOG_INDPRO"]
    return feat, macro_feat_cols


def _create_windows(df_scaled: pd.DataFrame, L: int, H: int, cond_cols):
    X, Y, C = [], [], []
    total_len = len(df_scaled)

    for start in range(total_len - L - H + 1):
        end_x = start + L
        end_y = end_x + H

        x = df_scaled.iloc[start:end_x].to_numpy(dtype=np.float32)     # (L,D)
        y = df_scaled.iloc[end_x:end_y].to_numpy(dtype=np.float32)     # (H,D)
        c = df_scaled.iloc[end_x - 1][cond_cols].to_numpy(dtype=np.float32)  # (cond_dim,)

        X.append(x)
        Y.append(y)
        C.append(c)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(Y, dtype=np.float32),
        np.asarray(C, dtype=np.float32),
    )


def _create_macro_windows(df_macro_scaled: pd.DataFrame, L: int, H: int):
    """
    macro-only windows:
      returns MACRO_X: (N, macro_dim, L)
    """
    MX = []
    total_len = len(df_macro_scaled)

    for start in range(total_len - L - H + 1):
        end_x = start + L
        m = df_macro_scaled.iloc[start:end_x].to_numpy(dtype=np.float32)  # (L,macro_dim)
        m = np.transpose(m, (1, 0))  # (macro_dim, L)
        MX.append(m)

    return np.asarray(MX, dtype=np.float32)


# -------------------------------
# Preprocess
# -------------------------------
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
    drop_semi_cols=("PMI",),            # semi쪽 PMI 제거 (macro PMI로 대체)
    capex_col: str = "CAPEX",
    fill_strategy: str = "ffill_bfill",  # "zero" or "ffill_bfill"
    macro_scaler_path: str | None = None,  # "macro_scaler.pkl" (optional)
):
    """
    Returns 9 values:
      X, Y, C, MACRO_X, scaler, df_raw, df_all_scaled, macro_feature_indices, macro_feat_cols
    """
    # 1) load
    df_semi = _read_csv_with_date_index(csv_path, encoding=semi_encoding)
    df_macro = _read_csv_with_date_index(macro_csv_path, encoding=macro_encoding)

    # 2) drop semi columns
    for col in drop_semi_cols:
        if col in df_semi.columns:
            df_semi = df_semi.drop(columns=[col])

    # 3) make macro features (6 cols) from macro_cols base
    df_macro_feat, macro_feat_cols = _make_macro_features(df_macro, macro_cols)

    # 4) merge by date
    df_all = df_semi.join(df_macro_feat, how="inner")
    df_raw = df_all.copy()

    # 5) numeric coercion (semi side too)
    df_all = _coerce_numeric_df(df_all)

    # 6) CAPEX log1p
    if capex_col in df_all.columns:
        cap = df_all[capex_col].to_numpy(dtype=float)
        cap = np.clip(cap, a_min=0.0, a_max=None)
        df_all[capex_col] = np.log1p(cap)

    # 7) missing
    for col in condition_raw_cols:
        if col not in df_all.columns:
            raise ValueError(f"condition column not found after merge: {col}")

    if fill_strategy == "ffill_bfill":
        df_all = df_all.ffill().bfill().fillna(0.0)
    elif fill_strategy == "zero":
        df_all = df_all.fillna(0.0)
    else:
        raise ValueError("fill_strategy must be 'ffill_bfill' or 'zero'")

    # 8) scale ALL features for main model / decoder space
    scaler = StandardScaler()
    df_all_scaled = pd.DataFrame(
        scaler.fit_transform(df_all),
        index=df_all.index,
        columns=df_all.columns,
    )

    # 9) macro indices in df_all_scaled (for debugging/figure, etc.)
    macro_feature_indices = [df_all_scaled.columns.get_loc(c) for c in macro_feat_cols]

    # 10) windows: X/Y/C from df_all_scaled
    X, Y, C = _create_windows(df_all_scaled, L, H, condition_raw_cols)

    # 11) MACRO_X: optionally apply macro_scaler.pkl (for macro encoder consistency)
    df_macro_feat_aligned = df_macro_feat.loc[df_all_scaled.index]  # same rows as merged period

    if macro_scaler_path is not None:
        if joblib is None:
            raise ImportError("joblib is required to load macro_scaler_path, but joblib import failed.")
        macro_scaler = joblib.load(macro_scaler_path)
        df_macro_scaled_for_encoder = pd.DataFrame(
            macro_scaler.transform(df_macro_feat_aligned),
            index=df_macro_feat_aligned.index,
            columns=df_macro_feat_aligned.columns,
        )
    else:
        # fallback: use the same scaling as df_all_scaled for those macro columns
        df_macro_scaled_for_encoder = df_all_scaled[macro_feat_cols].copy()

    MACRO_X = _create_macro_windows(df_macro_scaled_for_encoder, L, H)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("C shape:", C.shape)
    print("MACRO_X shape:", MACRO_X.shape)
    print("macro_feat_cols:", macro_feat_cols)
    print("macro_feature_indices (in df_all_scaled):", macro_feature_indices)

    return X, Y, C, MACRO_X, scaler, df_raw, df_all_scaled, macro_feature_indices, macro_feat_cols


# -------------------------------
# Training
# -------------------------------
def train_model(
    X_train, Y_train, C_train, MACRO_X_train,
    latent_dim, cond_dim, hidden,
    H_len, beta, lr, epochs, batch_size, device,
    macro_latent_dim=32,
    macro_hidden_dim=128,
    macro_encoder_path="macro_encoder.pth",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    out_dim = X_train.shape[-1]
    macro_input_dim = MACRO_X_train.shape[1]

    train_dataset = TimeSeriesDataset(X_train, Y_train, C_train, MACRO_X_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 1) frozen macro encoder
    macro_encoder = MacroEncoder(
        input_dim=macro_input_dim,
        hidden_dim=macro_hidden_dim,
        latent_dim=macro_latent_dim,
    ).to(device)
    macro_encoder.load_state_dict(torch.load(macro_encoder_path, map_location=device))
    macro_encoder.eval()
    for p in macro_encoder.parameters():
        p.requires_grad = False

    # 2) CT-VAE (TimeVAE)
    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H_len).to(device)
    prior = ConditionalPrior(cond_dim, macro_latent_dim, latent_dim, hidden).to(device)

    model = TimeVAE(encoder, decoder, prior, macro_encoder, latent_dim, beta).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("======== Training start ========")
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        num_batches = 0

        for x, y, c, mx in train_loader:
            x = x.to(device)
            y = y.to(device)
            c = c.to(device)
            mx = mx.to(device)  # (B, macro_dim, L)

            loss, recon, kl, _, _, _ = model(
                x, c, mx, y,
                use_prior_sampling_if_no_y=False
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_recon += float(recon.item())
            epoch_kl += float(kl.item())
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
# 1) Rolling Backtest (fixed model)
# ======================================================================
def rolling_backtest(
    model_path,
    X, Y, C, MACRO_X,
    latent_dim, cond_dim, hidden, H, beta,
    macro_latent_dim=32,
    macro_hidden_dim=128,
    device="cuda",
    macro_encoder_path="macro_encoder.pth",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    out_dim = X.shape[-1]
    macro_input_dim = MACRO_X.shape[1]

    macro_encoder = MacroEncoder(macro_input_dim, macro_hidden_dim, macro_latent_dim).to(device)
    macro_encoder.load_state_dict(torch.load(macro_encoder_path, map_location=device))
    macro_encoder.eval()
    for p in macro_encoder.parameters():
        p.requires_grad = False

    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H).to(device)
    prior = ConditionalPrior(cond_dim, macro_latent_dim, latent_dim, hidden).to(device)

    model = TimeVAE(encoder, decoder, prior, macro_encoder, latent_dim, beta).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mses = []
    with torch.no_grad():
        for t in range(len(X)):
            x = torch.tensor(X[t:t+1]).float().to(device)
            c = torch.tensor(C[t:t+1]).float().to(device)
            mx = torch.tensor(MACRO_X[t:t+1]).float().to(device)
            y_true = Y[t:t+1]

            mean, _, _, _, _, _ = model(
                x, c, mx,
                y=None,
                use_prior_sampling_if_no_y=True
            )
            y_pred = mean.cpu().numpy()
            mses.append(np.mean((y_pred - y_true) ** 2))

    return float(np.mean(mses))


# ======================================================================
# 2) Rolling Forward Test (retrain per anchor)
# ======================================================================
def rolling_forward_test(
    X, Y, C, MACRO_X,
    latent_dim, cond_dim, hidden, H,
    beta, lr, epochs, batch_size,
    macro_latent_dim=32,
    macro_hidden_dim=128,
    device="cuda",
    macro_encoder_path="macro_encoder.pth",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    N = len(X)

    mse_list = []

    for anchor in range(1, N - 1):
        print(f"[Rolling-Forward] Anchor {anchor}/{N}")

        X_train = X[:anchor]
        Y_train = Y[:anchor]
        C_train = C[:anchor]
        MX_train = MACRO_X[:anchor]

        X_test = X[anchor:anchor + 1]
        Y_test = Y[anchor:anchor + 1]
        C_test = C[anchor:anchor + 1]
        MX_test = MACRO_X[anchor:anchor + 1]

        model = train_model(
            X_train, Y_train, C_train, MX_train,
            latent_dim, cond_dim, hidden,
            H, beta, lr, epochs, batch_size, device,
            macro_latent_dim=macro_latent_dim,
            macro_hidden_dim=macro_hidden_dim,
            macro_encoder_path=macro_encoder_path,
        )

        with torch.no_grad():
            x = torch.tensor(X_test).float().to(device)
            c = torch.tensor(C_test).float().to(device)
            mx = torch.tensor(MX_test).float().to(device)

            mean, _, _, _, _, _ = model(
                x, c, mx,
                y=None,
                use_prior_sampling_if_no_y=True
            )
            y_pred = mean.cpu().numpy()

        mse_list.append(np.mean((y_pred - Y_test) ** 2))

    return float(np.mean(mse_list))
