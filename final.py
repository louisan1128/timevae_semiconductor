###############################################
#  CT-VAE for Semiconductor Cycles 
###############################################

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import StudentT
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ===========================
#  Hyperparameters
# ===========================
L = 36          # 입력 시퀀스 길이
H = 12          # 예측 시퀀스 길이
LATENT_DIM = 32
COND_DIM = 5    # [Exchange, CAPEX, PMI, CLI, ISM]
HIDDEN = 128
POLY_ORDER = 2  # trend 차수 (0,1,2)
N_FOURIER = 3   # seasonality용 Fourier 개수
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-3
BETA = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# condition 컬럼 이름 (data.csv 기준)
condition_raw_cols = [
    "Exchange Rate",
    "CAPEX",
    "Global Manufacturing PMI",
    "OECD CLI",
    "U.S. ISM Manufacturing New Orders Index",
]




#  Encoder: ConditionLayer + TCN
from torch.nn.utils import weight_norm


class ConditionLayer(nn.Module):
    def __init__(self, c_dim, h_dim):
        super().__init__()
        self.fc1 = nn.Linear(c_dim, h_dim)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)

    def forward(self, c):
        x = self.fc1(c)
        x = self.act(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs,
                 kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1)
            if n_inputs != n_outputs
            else None
        )
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels,
                 kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(
                TemporalBlock(
                    n_inputs=in_ch,
                    n_outputs=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class FiLM(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta  = nn.Linear(cond_dim, hidden_dim)

    def forward(self, h, c):
        """
        h: (B, hidden, L)
        c: (B, hidden)
        """
        γ = self.gamma(c).unsqueeze(-1)  # (B, hidden, 1)
        β = self.beta(c).unsqueeze(-1)   # (B, hidden, 1)
        return γ * h + β

class FiLM_TCN(nn.Module):
    def __init__(self, hidden_dim, tcn_layers=4, kernel_size=3, dropout=0.15):
        super().__init__()

        self.layers = nn.ModuleList()
        self.films  = nn.ModuleList()

        for i in range(tcn_layers):
            dilation = 2 ** i
            self.layers.append(
                TemporalBlock(
                    n_inputs=hidden_dim,
                    n_outputs=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )

            # FiLM per layer
            self.films.append(
                FiLM(cond_dim=hidden_dim, hidden_dim=hidden_dim)
            )

    def forward(self, h, c_embed):
        """
        h: (B, hidden, L)
        c_embed: (B, hidden)
        """
        # DO NOT unsqueeze here
        # c_embed is (B, hidden)

        for film, block in zip(self.films, self.layers):
            h = film(h, c_embed)   # FiLM will unsqueeze inside
            h = block(h)
        return h



class Encoder(nn.Module):
    def __init__(self, x_dim, c_dim, h_dim, z_dim,
                 tcn_layers=4, kernel_size=3):
        super().__init__()

        # 1) condition embedding
        self.condition_layer = ConditionLayer(c_dim, h_dim)

        # 2) input projection
        self.input_proj = nn.Conv1d(x_dim, h_dim, kernel_size=1)

        # 3) FiLM + TCN layered
        self.tcn_film = FiLM_TCN(
            hidden_dim=h_dim,
            tcn_layers=tcn_layers,
            kernel_size=kernel_size,
            dropout=0.15
        )


        # 4) latent heads
        self.mu_layer = nn.Linear(h_dim, z_dim)
        self.logvar_layer = nn.Linear(h_dim, z_dim)

    def forward(self, x, c):
        """
        x: (B, L, x_dim)
        c: (B, c_dim)
        """

        # input projection
        x = x.permute(0, 2, 1)
        h = self.input_proj(x)        # (B, hidden, L)

        # condition embedding
        c_embed = self.condition_layer(c)  # (B, hidden)

        # Layer-wise FiLM + TCN
        h = self.tcn_film(h, c_embed)

        # last timestep
        h_last = h[:, :, -1]

        mu = self.mu_layer(h_last)
        logvar = self.logvar_layer(h_last)
        return mu, logvar



# Decoder: Trend + Seasonality + Residual + Student-t
class Decoder(nn.Module):
    """
    Time-dependent Decoder with:
      - Polynomial Trend
      - Fourier Seasonality
      - GRU Residual
      - Student-t likelihood
    """

    def __init__(
        self,
        latent_dim,
        cond_dim,
        out_dim,
        hidden,
        H,
        poly_order=2,
        n_fourier=3,
    ):
        super().__init__()
        self.H = H
        self.out_dim = out_dim
        self.hidden = hidden
        self.poly_order = poly_order
        self.n_fourier = n_fourier

        # 1) context from z, c
        self.fc_context = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # 2) Trend coefficients θ_tr (B, D, P+1)
        self.trend_head = nn.Linear(hidden, out_dim * (poly_order + 1))

        # 3) Seasonality coefficients θ_sin, θ_cos (B, D, K)
        self.season_sin_head = nn.Linear(hidden, out_dim * n_fourier)
        self.season_cos_head = nn.Linear(hidden, out_dim * n_fourier)

        # 4) Residual RNN
        self.rnn = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            batch_first=True,
        )
        self.rnn_out = nn.Linear(hidden, out_dim)

        # 5) Student-t parameters
        self.fc_scale = nn.Linear(hidden, H * out_dim)
        self.fc_df = nn.Linear(hidden, H * out_dim)

    def _time_grid(self, device):
        # r_t ∈ [0,1], t_angle ∈ [0,2π]
        r = torch.linspace(0.0, 1.0, self.H, device=device)
        t_angle = torch.linspace(0.0, 2 * math.pi, self.H, device=device)
        return r, t_angle

    def forward(self, z, c):
        """
        z: (B, latent_dim)
        c: (B, cond_dim)
        return:
          mean: (B, H, D)
          dist: StudentT(df, loc, scale) over (B,H,D)
        """
        device = z.device
        B = z.size(0)
        D = self.out_dim

        x = torch.cat([z, c], dim=-1)     # (B, latent+cond)
        h = self.fc_context(x)           # (B, hidden)

        # ----- 시간 grid -----
        r, t_angle = self._time_grid(device)  # (H,), (H,)
        powers = torch.stack(
            [r**p for p in range(self.poly_order + 1)],
            dim=-1
        )  # (H, P+1)

        ks = torch.arange(1, self.n_fourier + 1, device=device).float()
        angles = t_angle.unsqueeze(-1) * ks.unsqueeze(0)  # (H,K)
        sin_basis = torch.sin(angles)  # (H,K)
        cos_basis = torch.cos(angles)  # (H,K)

        # ----- Trend -----
        trend_theta = self.trend_head(h)  # (B, D*(P+1))
        trend_theta = trend_theta.view(B, D, self.poly_order + 1)  # (B,D,P+1)

        powers_expanded = powers.unsqueeze(0).unsqueeze(0)         # (1,1,H,P+1)
        trend_theta_expanded = trend_theta.unsqueeze(2)            # (B,D,1,P+1)

        trend = (trend_theta_expanded * powers_expanded).sum(-1)   # (B,D,H)
        trend = trend.permute(0, 2, 1)                             # (B,H,D)

        # ----- Seasonality -----
        season_sin = self.season_sin_head(h).view(B, D, self.n_fourier)  # (B,D,K)
        season_cos = self.season_cos_head(h).view(B, D, self.n_fourier)

        sin_b = sin_basis.unsqueeze(0).unsqueeze(0)   # (1,1,H,K)
        cos_b = cos_basis.unsqueeze(0).unsqueeze(0)   # (1,1,H,K)

        sin_theta = season_sin.unsqueeze(2)           # (B,D,1,K)
        cos_theta = season_cos.unsqueeze(2)           # (B,D,1,K)

        sea = (sin_theta * sin_b + cos_theta * cos_b).sum(-1)  # (B,D,H)
        sea = sea.permute(0, 2, 1)                             # (B,H,D)

        # ----- Residual RNN -----
        rnn_input = h.unsqueeze(1).repeat(1, self.H, 1)  # (B,H,hidden)
        rnn_out, _ = self.rnn(rnn_input)
        residual = self.rnn_out(rnn_out)                 # (B,H,D)

        # ----- Final mean -----
        mean = trend + sea + residual                    # (B,H,D)

        # ----- Student-t params -----
        scale = F.softplus(self.fc_scale(h)).view(B, self.H, D) + 1e-4
        df = F.softplus(self.fc_df(h)).view(B, self.H, D) + 2.0

        dist = StudentT(df, loc=mean, scale=scale)
        return mean, dist




# Conditional Prior p(z | c)
class ConditionalPrior(nn.Module):
    """
    p(z | c) = N(mu_p(c), diag(sigma_p(c)^2))
    """

    def __init__(self, cond_dim, latent_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, latent_dim)
        self.logvar_head = nn.Linear(hidden, latent_dim)

    def forward(self, c):
        h = self.net(c)
        mu_p = self.mu_head(h)
        logvar_p = self.logvar_head(h)
        return mu_p, logvar_p


###############################################
# TimeVAE (CT-VAE with conditional prior)
###############################################

class TimeVAE(nn.Module):
    def __init__(self, encoder, decoder, prior, latent_dim, beta=1.0):
        super().__init__()
        self.encoder = encoder   # q(z|x,c)
        self.decoder = decoder   # p(x|z,c)
        self.prior = prior       # p(z|c)
        self.latent_dim = latent_dim
        self.beta = beta

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)
        return mu + eps * std

    def kl_gaussian(self, mu_q, logvar_q, mu_p, logvar_p):
        """
        KL( N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2) ), diag 가정
        """
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        term = (
            logvar_p - logvar_q
            + (var_q + (mu_q - mu_p) ** 2) / var_p
            - 1.0
        )
        return 0.5 * term.sum(dim=-1).mean()

    def forward(self, x, c, y=None, use_prior_sampling_if_no_y=True):
        """
        x: (B, L, D)
        c: (B, cond_dim)
        y: (B, H, D) or None
        """
        # posterior q(z|x,c)
        mu_q, logvar_q = self.encoder(x, c)

        # prior p(z|c)
        mu_p, logvar_p = self.prior(c)

        # inference-only 모드 (scenario generation)
        if (y is None) and use_prior_sampling_if_no_y:
            z = self.reparameterize(mu_p, logvar_p)
            mean, dist = self.decoder(z, c)
            return mean, z, (mu_p, logvar_p)

        # 학습 모드: q에서 샘플링
        z = self.reparameterize(mu_q, logvar_q)
        mean, dist = self.decoder(z, c)

        # Student-t NLL
        log_prob = dist.log_prob(y)  # (B,H,D)
        recon_loss = -log_prob.mean()

        # KL(q || p)
        kl_loss = self.kl_gaussian(mu_q, logvar_q, mu_p, logvar_p)

        loss = recon_loss + self.beta * kl_loss
        return loss, recon_loss, kl_loss, mean, z


###############################################
# Dataset / Preprocessing
###############################################

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, c):
        self.x = x
        self.y = y
        self.c = c

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.c[idx]


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


def preprocess(csv_path="data.csv"):
    df = pd.read_csv(csv_path, encoding="latin1")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    df_raw = df.copy()
    # 수치 변환
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df['CAPEX'] = np.log1p(df['CAPEX'])

    df[condition_raw_cols] = df[condition_raw_cols].fillna(0)

    # StandardScaler
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns,
    )

    X, Y, C = create_dataset(df_scaled, L, H, condition_raw_cols)

    print("X shape:", X.shape)  # (N,L,D)
    print("Y shape:", Y.shape)  # (N,H,D)
    print("C shape:", C.shape)  # (N,cond_dim)

    return X, Y, C, scaler, df_raw


###############################################
# Training / Evaluation / Scenario Forecast
###############################################

def train_model(
    x_train,
    y_train,
    c_train,
    latent_dim=LATENT_DIM,
    cond_dim=COND_DIM,
    hidden=HIDDEN,
    H_len=H,
    beta=BETA,
    lr=LR,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    device=DEVICE,
):
    device = torch.device(device)

    out_dim = x_train.shape[-1]

    train_dataset = TimeSeriesDataset(x_train, y_train, c_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    encoder = Encoder(
        x_dim=out_dim,
        c_dim=cond_dim,
        h_dim=hidden,
        z_dim=latent_dim,
    ).to(device)

    decoder = Decoder(
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        out_dim=out_dim,
        hidden=hidden,
        H=H_len,
        poly_order=POLY_ORDER,
        n_fourier=N_FOURIER,
    ).to(device)

    prior = ConditionalPrior(
        cond_dim=cond_dim,
        latent_dim=latent_dim,
        hidden=hidden,
    ).to(device)

    model = TimeVAE(
        encoder=encoder,
        decoder=decoder,
        prior=prior,
        latent_dim=latent_dim,
        beta=beta,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("======== Training start ========")
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        num_batches = 0

        for x_batch, y_batch, c_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            c_batch = c_batch.to(device)

            loss, recon, kl, mean, z = model(x_batch, c_batch, y_batch)

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


def evaluate_model(
    model_path,
    X,
    Y,
    C,
    latent_dim=LATENT_DIM,
    cond_dim=COND_DIM,
    hidden=HIDDEN,
    H_len=H,
    beta=BETA,
    device=DEVICE,
):
    device = torch.device(device)
    out_dim = X.shape[-1]

    encoder = Encoder(
        x_dim=out_dim,
        c_dim=cond_dim,
        h_dim=hidden,
        z_dim=latent_dim,
    ).to(device)

    decoder = Decoder(
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        out_dim=out_dim,
        hidden=hidden,
        H=H_len,
        poly_order=POLY_ORDER,
        n_fourier=N_FOURIER,
    ).to(device)

    prior = ConditionalPrior(
        cond_dim=cond_dim,
        latent_dim=latent_dim,
        hidden=hidden,
    ).to(device)

    model = TimeVAE(
        encoder=encoder,
        decoder=decoder,
        prior=prior,
        latent_dim=latent_dim,
        beta=beta,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds = []
    trues = []

    with torch.no_grad():
        for i in range(len(X)):
            x = torch.tensor(X[i:i+1], dtype=torch.float32).to(device)
            c = torch.tensor(C[i:i+1], dtype=torch.float32).to(device)
            y = torch.tensor(Y[i:i+1], dtype=torch.float32).to(device)

            # 학습 모드와 동일하게 posterior를 써서 reconstruction
            loss, recon, kl, mean, z = model(x, c, y)
            preds.append(mean.cpu().numpy())
            trues.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)  # (N,H,D)
    trues = np.concatenate(trues, axis=0)  # (N,H,D)

    mse = np.mean((preds - trues) ** 2)
    print("===================================")
    print(f"Evaluation MSE (posterior recon): {mse:.6f}")
    print("===================================")
    return preds, trues, mse


def rolling_forward_backtest(X, Y, C):
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

    for anchor in range(L, N - H):
        print(f"[Rolling-Forward] Anchor = {anchor}/{N}")

        # ----------------------------
        # 1) Expanding Window Train Split
        # ----------------------------
        X_train = X[:anchor]
        Y_train = Y[:anchor]
        C_train = C[:anchor]

        # Test one step (anchor)
        X_test = X[anchor:anchor+1]
        Y_test = Y[anchor:anchor+1]
        C_test = C[anchor:anchor+1]

        # ----------------------------
        # 2) 모델 재학습
        # ----------------------------
        model = train_model(
            X_train,
            Y_train,
            C_train,
            epochs=50  # 너무 느리면 줄여도 됨
        )

        # ----------------------------
        # 3) Forecast (True Forecast Mode)
        # ----------------------------
        with torch.no_grad():
            x = torch.tensor(X_test, dtype=torch.float32).to(device)
            c = torch.tensor(C_test, dtype=torch.float32).to(device)

            mean, _, _ = model(
                x, c,
                y=None,
                use_prior_sampling_if_no_y=True
            )

            y_pred = mean.cpu().numpy()

        # ----------------------------
        # 4) MSE 계산
        # ----------------------------
        mse = np.mean((y_pred - Y_test)**2)
        mse_list.append(mse)

    # 전체 anchor 평균 MSE
    return np.mean(mse_list)

def rolling_backtest(model_path, X, Y, C):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 리로드
    out_dim = X.shape[-1]

    encoder = Encoder(
        x_dim=out_dim, c_dim=COND_DIM, h_dim=HIDDEN, z_dim=LATENT_DIM
    ).to(device)

    decoder = Decoder(
        latent_dim=LATENT_DIM, cond_dim=COND_DIM, out_dim=out_dim,
        hidden=HIDDEN, H=H, poly_order=POLY_ORDER, n_fourier=N_FOURIER
    ).to(device)

    prior = ConditionalPrior(
        cond_dim=COND_DIM, latent_dim=LATENT_DIM, hidden=HIDDEN
    ).to(device)

    model = TimeVAE(
        encoder=encoder, decoder=decoder, prior=prior,
        latent_dim=LATENT_DIM, beta=BETA
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mses = []

    for t in range(len(X)):
        x = torch.tensor(X[t:t+1], dtype=torch.float32).to(device)
        c = torch.tensor(C[t:t+1], dtype=torch.float32).to(device)
        y_true = Y[t:t+1]   # numpy

        # forecast mode
        with torch.no_grad():
            y_pred, _, _ = model(x, c, y=None, use_prior_sampling_if_no_y=True)
            y_pred = y_pred.cpu().numpy()

        mse_t = np.mean((y_pred - y_true)**2)
        mses.append(mse_t)

    return np.mean(mses)

def scenario_predict_local(
    model_path,
    X_last,            # (L, D) 마지막 입력 시퀀스 (scaled)
    cond_true,         # (cond_dim,) 마지막 window에서 실제 condition (C[-1])
    cond_scenario,     # (cond_dim,) 시나리오 condition (scaled)
    latent_dim=LATENT_DIM,
    cond_dim=COND_DIM,
    hidden=HIDDEN,
    H_len=H,
    beta=BETA,
    device=DEVICE,
    num_samples=20,
    z_shrink=0.1,      # 🔥 z 변동 폭 (0.1이면 truth 근처, 크면 시나리오 더 퍼짐)
):
    """
    posterior q(z|x,c_true)를 anchor로 쓰고,
    그 주변에서만 작은 noise를 주어 scenario를 여러 개 샘플링한다.
    """

    device = torch.device(device)
    out_dim = X_last.shape[-1]

    # ----- 모델 로드 (evaluate_model과 동일한 방식) -----
    encoder = Encoder(
        x_dim=out_dim,
        c_dim=cond_dim,
        h_dim=hidden,
        z_dim=latent_dim,
    ).to(device)

    decoder = Decoder(
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        out_dim=out_dim,
        hidden=hidden,
        H=H_len,
        poly_order=POLY_ORDER,
        n_fourier=N_FOURIER,
    ).to(device)

    prior = ConditionalPrior(
        cond_dim=cond_dim,
        latent_dim=latent_dim,
        hidden=hidden,
    ).to(device)

    model = TimeVAE(
        encoder=encoder,
        decoder=decoder,
        prior=prior,
        latent_dim=latent_dim,
        beta=beta,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ----- 텐서로 변환 -----
    X_tensor = torch.tensor(X_last, dtype=torch.float32).unsqueeze(0).to(device)      # (1,L,D)
    C_true   = torch.tensor(cond_true, dtype=torch.float32).unsqueeze(0).to(device)   # (1,cond_dim)
    C_scen   = torch.tensor(cond_scenario, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        # 1) posterior q(z | x_last, c_true) 계산
        mu_q, logvar_q = model.encoder(X_tensor, C_true)        # (1, latent_dim)

        # 2) posterior mean을 anchor로 사용
        z_base = mu_q                                           # (1, latent_dim)

        # 3) z 분산을 줄여서 truth 근처에서만 샘플
        std_q = torch.exp(0.5 * logvar_q) * z_shrink            # shrink factor 적용

        samples = []
        for _ in range(num_samples):
            eps = torch.randn_like(std_q)
            z_k = z_base + eps * std_q                          # anchored latent

            mean, _ = model.decoder(z_k, C_scen)                # (1,H,D)
            samples.append(mean.cpu().numpy()[0])

    samples = np.stack(samples, axis=0)  # (num_samples, H, D)
    return samples


def plot_fanchart(
    true_seq,
    pred_seq,
    scenario_samples,
    scenario_cond_raw,
    scenario_cond_scaled,
    last_truth_raw,
    condition_raw_cols,
    feature_index=0
):
    import numpy as np
    import matplotlib.pyplot as plt

    scenario_samples = np.array(scenario_samples)
    num_samples, H, D = scenario_samples.shape

    # 분위수
    lower = np.percentile(scenario_samples[:, :, feature_index], 10, axis=0)
    median = np.percentile(scenario_samples[:, :, feature_index], 50, axis=0)
    upper = np.percentile(scenario_samples[:, :, feature_index], 90, axis=0)

    # -------------------------------
    # Fan chart 그림
    # -------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(median, color="red", label="Scenario Median")
    plt.plot(pred_seq[:, feature_index], color="blue", label="Model Prediction")
    plt.plot(true_seq[:, feature_index], color="black", label="Ground Truth")

    plt.title(f"Fan Chart: Export Index")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # Condition 비교 출력 (확실하게 보이도록 정렬)
    # -------------------------------
    print("\n================ Condition Comparison ================")
    print(f"{'Feature Name':40s}{'Truth':>12s}{'Scenario':>12s}")
    print("-" * 64)

    for col in condition_raw_cols:
        truth_val = float(str(last_truth_raw[col]).replace(",", ""))
        scenario_val = float(scenario_cond_raw[col])

        changed = "  <-- CHANGED" if abs(truth_val - scenario_val) > 1e-6 else ""

        print(f"{col:40s}{truth_val:12.2f}{scenario_val:12.2f}{changed}")

    print("=" * 64)

    # 스케일된 값도 보여줌
    print("\nScaled Scenario Condition (z-score):")
    for i, col in enumerate(condition_raw_cols):
        print(f"{col:40s} {scenario_cond_scaled[i]: .4f}")


###############################################
# Plot Helper
###############################################

def plot_forecast(true, pred, title="Forecast vs Real", feature_index=0):
    plt.figure(figsize=(8, 4))
    plt.plot(true[:, feature_index], label="Real", color="black")
    plt.plot(pred[:, feature_index], label="Prediction", color="red")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


###############################################
# Main
###############################################

if __name__ == "__main__":
    # 1) 전처리
    X, Y, C, scaler, df_raw = preprocess("data.csv")

    # 2) 학습
    model = train_model(X, Y, C)

    mse_rolling = rolling_backtest(
        model_path="timevae_ctvae_prior.pth",
        X=X,
        Y=Y,
        C=C
    )

    print("Rolling Backtest MSE:", mse_rolling)
    # 3) 평가 (posterior reconstruction 기준)
    preds, trues, mse = evaluate_model("timevae_ctvae_prior.pth", X, Y, C)

    plot_forecast(trues[1], preds[1], title="Example Prediction", feature_index=0)



    last_truth_raw = df_raw.iloc[-1]

    # 3) condition column 리스트 (이미 preprocess 안에 있음)
    condition_raw_cols = [
        "Exchange Rate",
        "CAPEX",
        "Global Manufacturing PMI",
        "OECD CLI",
        "U.S. ISM Manufacturing New Orders Index",
    ]

    # 4) 네가 설정하고 싶은 condition raw 값
    scenario_cond_raw = {
        "Exchange Rate": 1100,
        "CAPEX": 0.0,
        "Global Manufacturing PMI": 52.90,
        "OECD CLI": 100.43,
        "U.S. ISM Manufacturing New Orders Index": 49.40,
    }

    # 5) full_raw 만들기 (Export/DRAM 등은 truth 유지)
    full_raw = [
        scenario_cond_raw[col] if col in scenario_cond_raw else last_truth_raw[col]
        for col in df_raw.columns
    ]

    # 6) scaled 변환
    custom_scaled_full = scaler.transform([full_raw])[0]

    scenario_cond_scaled = np.array([
        custom_scaled_full[df_raw.columns.get_loc(col)]
        for col in condition_raw_cols
    ])
    # -----------------------------
    # (5) Scenario Sampling 생성
    # -----------------------------
    cond_true_last = C[-1]               # (cond_dim,)

    # 우리가 만든 시나리오 condition (scaled) = scenario_cond_scaled
    cond_scen = scenario_cond_scaled      # (cond_dim,)
    
    scenario_samples = scenario_predict_local(
        model_path="timevae_ctvae_prior.pth",
        X_last=X[-1],                     # 마지막 36개월 입력 (scaled)
        cond_true=cond_true_last,         # 실제 마지막 조건
        cond_scenario=cond_scen,          # what-if 조건
        num_samples=50,                   
        z_shrink=0.1,                     
    )

    print("Scenario samples shape:", scenario_samples.shape)
    # → (50, H, D)

    # -----------------------------
    # (6) Fan Chart Plotting
    # -----------------------------
    plot_fanchart(
        true_seq=trues[1],
        pred_seq=preds[1],
        scenario_samples=scenario_samples - 3,
        scenario_cond_raw=scenario_cond_raw,
        scenario_cond_scaled=scenario_cond_scaled,
        last_truth_raw=last_truth_raw,
        condition_raw_cols=condition_raw_cols,
        feature_index=0
    )