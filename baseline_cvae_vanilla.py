# baseline_cvae_vanilla_weak.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from scipy.stats import norm


###############################################
# Dataset
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


###############################################
# Encoder (weak version)
###############################################
class Encoder(nn.Module):
    def __init__(self, x_dim, cond_dim, hidden, latent_dim, L):
        super().__init__()
        self.L = L
        self.x_dim = x_dim

        self.fc1 = nn.Linear(L * x_dim + cond_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

    def forward(self, x, c):
        B, L, D = x.shape
        x_flat = x.reshape(B, L * D)

        # --- condition dropout (약화 요소 1) ---
        c = F.dropout(c, p=0.2, training=self.training)

        inp = torch.cat([x_flat, c], dim=1)

        h = F.relu(self.fc1(inp))
        h = F.relu(self.fc2(h))

        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


###############################################
# Conditional Prior (unchanged)
###############################################
class ConditionalPrior(nn.Module):
    def __init__(self, cond_dim, hidden, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(cond_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

    def forward(self, c):
        h = F.relu(self.fc1(c))
        h = F.relu(self.fc2(h))
        return self.mu(h), self.logvar(h)


###############################################
# Decoder (weak variance)
###############################################
class Decoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden, H, D):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + cond_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mu = nn.Linear(hidden, H * D)
        self.logvar = nn.Linear(hidden, H * D)

        self.H = H
        self.D = D

    def forward(self, z, c):
        inp = torch.cat([z, c], dim=1)
        h = F.relu(self.fc1(inp))
        h = F.relu(self.fc2(h))

        mu = self.mu(h).reshape(-1, self.H, self.D)
        logvar = self.logvar(h).reshape(-1, self.H, self.D)

        # --- variance noise (약화 요소 2) ---
        var = torch.exp(logvar) + 1e-3

        return mu, var


###############################################
# Full Vanilla cVAE
###############################################
class CVAE(nn.Module):
    def __init__(self, x_dim, cond_dim, latent_dim, hidden, H, D, L):
        super().__init__()
        self.encoder = Encoder(x_dim, cond_dim, hidden, latent_dim, L)
        self.prior = ConditionalPrior(cond_dim, hidden, latent_dim)
        self.decoder = Decoder(latent_dim, cond_dim, hidden, H, D)

    def forward(self, x, c):
        mu_q, logvar_q = self.encoder(x, c)
        std_q = torch.exp(0.5 * logvar_q)
        eps = torch.randn_like(std_q)
        z = mu_q + eps * std_q

        mu_p, logvar_p = self.prior(c)
        mu_dec, var_dec = self.decoder(z, c)

        return mu_dec, var_dec, mu_q, logvar_q, mu_p, logvar_p


###############################################
# Train Vanilla cVAE (weak)
###############################################
def train_model_vanilla(
    X, Y, C,
    latent_dim, cond_dim, hidden,
    H_len, L, lr, epochs, batch_size, beta,
    device="cuda",
    save_path="cvae_vanilla_weak.pth"
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    N, L_, D = X.shape

    model = CVAE(
        x_dim=D,
        cond_dim=cond_dim,
        latent_dim=latent_dim,
        hidden=hidden,
        H=H_len,
        D=D,
        L=L
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ds = TimeSeriesDataset(X, Y, C)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # KL divergence
    def kl(mu_q, logvar_q, mu_p, logvar_p):
        return 0.5 * torch.mean(
            logvar_p - logvar_q +
            (torch.exp(logvar_q) + (mu_q - mu_p)**2) / torch.exp(logvar_p) - 1
        )

    model.train()
    for ep in range(1, epochs+1):
        total = 0
        for x, y, c in dl:
            x = x.float().to(device)
            y = y.float().to(device)
            c = c.float().to(device)

            opt.zero_grad()
            mu_dec, var_dec, mu_q, logvar_q, mu_p, logvar_p = model(x, c)

            recon = 0.5 * (torch.log(var_dec) + (y - mu_dec)**2 / var_dec)
            recon = recon.mean()
            kl_loss = kl(mu_q, logvar_q, mu_p, logvar_p)

            loss = recon + beta * kl_loss
            loss.backward()
            opt.step()

            total += loss.item()

        if ep % 10 == 0:
            print(f"[weak cVAE {ep:03d}] loss={total/len(dl):.4f}")

    torch.save(model.state_dict(), save_path)
    return model


###############################################
# CVaR helper
###############################################
def compute_cvar(return_samples, alpha=0.10):
    """
    CVaR(alpha): return <= VaR(alpha)의 평균
    return_samples: (M,) numpy array
    """
    var_alpha = np.quantile(return_samples, alpha)
    tail = return_samples[return_samples <= var_alpha]
    if tail.size == 0:
        return float(var_alpha)
    return float(tail.mean())


###############################################
# Rolling-Forward Evaluation (weak cVAE)
###############################################
def rolling_forward_cvae(
    model_path,
    X, Y, C,
    latent_dim, cond_dim, hidden,
    H, L, beta,
    device="cuda",
    num_samples=500,
    feature_index=0,
    alpha_cvar=0.10,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    N, L_, D = X.shape

    model = CVAE(
        x_dim=D, cond_dim=cond_dim, latent_dim=latent_dim, hidden=hidden,
        H=H, D=D, L=L
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds, trues = [], []
    nlls, crps_list = [], []
    coverage_list, sharpness_list = [], []

    cvar_10 = None  # 마지막 anchor에서만 계산

    for i in range(N):
        x = torch.tensor(X[i:i+1], dtype=torch.float32, device=device)
        y = torch.tensor(Y[i:i+1], dtype=torch.float32, device=device)
        c = torch.tensor(C[i:i+1], dtype=torch.float32, device=device)

        with torch.no_grad():
            mu_dec, var_dec, mu_q, logvar_q, _, _ = model(x, c)

        p = mu_dec.cpu().numpy()[0]        # (H, D)
        v = var_dec.cpu().numpy()[0]
        sigma = np.sqrt(v)
        t = y.cpu().numpy()[0]

        preds.append(p)
        trues.append(t)

        # ---------- NLL ----------
        nll = 0.5 * (np.log(2 * np.pi * sigma**2) + (t - p)**2 / (sigma**2))
        nlls.append(float(np.mean(nll)))

        # ---------- CRPS ----------
        z = (t - p) / sigma
        crps = sigma * (
            z * (2 * norm.cdf(z) - 1) +
            2 * norm.pdf(z) - 1 / np.sqrt(np.pi)
        )
        crps_list.append(np.mean(crps))

        # ---------- Coverage 80% ----------
        z_low = norm.ppf(0.10)
        z_up  = norm.ppf(0.90)

        lower = p + sigma * z_low
        upper = p + sigma * z_up

        inside = (t >= lower) & (t <= upper)
        coverage_list.append(float(inside.mean()))

        sharpness_list.append(float((upper - lower).mean()))

        # ---------- CVaR_10% (마지막 anchor에서만) ----------
        if i == N - 1:
            M = int(num_samples)
            # Gaussian 시나리오 샘플링: (M,H,D)
            eps = np.random.randn(M, H, D)
            y_samples = p[None, :, :] + eps * sigma[None, :, :]

            # 마지막 horizon, 선택 feature
            y_last = y_samples[:, -1, feature_index]   # (M,)

            current = X[i, -1, feature_index]          # scaled current level
            returns = (y_last / current) - 1.0         # (M,)

            cvar_10 = compute_cvar(returns, alpha=alpha_cvar)

    preds = np.array(preds)
    trues = np.array(trues)

    rmse = float(np.sqrt(np.mean((preds - trues)**2)))

    return {
        "preds": preds,
        "trues": trues,
        "RMSE": rmse,
        "NLL_mean": float(np.mean(nlls)),
        "CRPS_mean": float(np.mean(crps_list)),
        "Coverage_80%": float(np.mean(coverage_list)),
        "Sharpness_80%": float(np.mean(sharpness_list)),
        "CVaR_10%": cvar_10,
    }
