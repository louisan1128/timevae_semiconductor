# baseline_cvae_vanilla.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.distributions import StudentT
from scipy.stats import norm


###############################################
# Dataset (same as TimeSeriesDataset)
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
#  Encoder (no macro)
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
        inp = torch.cat([x_flat, c], dim=1)
        h = F.relu(self.fc1(inp))
        h = F.relu(self.fc2(h))
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


###############################################
#  Conditional Prior (no macro)
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
#  Decoder (Student-t)
###############################################
class Decoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden, H, D):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + cond_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mu = nn.Linear(hidden, H * D)
        self.log_scale = nn.Linear(hidden, H * D)
        self.df = nn.Parameter(torch.tensor(5.0))
        self.H = H
        self.D = D

    def forward(self, z, c):
        inp = torch.cat([z, c], dim=1)
        h = F.relu(self.fc1(inp))
        h = F.relu(self.fc2(h))
        mu = self.mu(h).reshape(-1, self.H, self.D)
        scale = torch.exp(self.log_scale(h)).reshape(-1, self.H, self.D) + 1e-6
        dist = StudentT(df=self.df, loc=mu, scale=scale)
        return mu, dist


###############################################
#  Full Vanilla cVAE
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
        mu_dec, dist = self.decoder(z, c)
        return mu_dec, dist, mu_q, logvar_q, mu_p, logvar_p


###############################################
#  Train Vanilla cVAE
###############################################
def train_model_vanilla(
    X, Y, C,
    latent_dim, cond_dim, hidden,
    H_len, L, lr, epochs, batch_size, beta,
    device="cuda",
    save_path="cvae_vanilla.pth"
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
            mu_dec, dist, mu_q, logvar_q, mu_p, logvar_p = model(x, c)

            recon = -dist.log_prob(y).mean()
            kl_loss = kl(mu_q, logvar_q, mu_p, logvar_p)
            loss = recon + beta * kl_loss

            loss.backward()
            opt.step()
            total += loss.item()

        if ep % 10 == 0:
            print(f"[vanilla cVAE {ep:03d}] loss={total/len(dl):.4f}")

    torch.save(model.state_dict(), save_path)
    return model


###############################################
#  Rolling-Forward eval (same style as LSTM/ARIMA)
###############################################
def rolling_forward_cvae(
    model_path,
    X, Y, C,
    latent_dim, cond_dim, hidden,
    H, L, beta,
    device="cuda"
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
    nlls = []
    crps_list = []

    for i in range(N):
        x = torch.tensor(X[i:i+1], dtype=torch.float32, device=device)
        y = torch.tensor(Y[i:i+1], dtype=torch.float32, device=device)
        c = torch.tensor(C[i:i+1], dtype=torch.float32, device=device)

        with torch.no_grad():
            mu_dec, dist, _, _, _, _ = model(x, c)

        p = mu_dec.cpu().numpy()[0]        # (H,D)
        t = y.cpu().numpy()[0]             # (H,D)
        preds.append(p)
        trues.append(t)

        # NLL
        nlls.append(float(-dist.log_prob(y).mean().cpu().item()))

        # CRPS (Gaussian approx)
        sigma = np.std(p - t) + 1e-6
        z = (t - p) / sigma
        crps = sigma * (z*(2*norm.cdf(z)-1) + 2*norm.pdf(z) - 1/np.sqrt(np.pi))
        crps_list.append(np.mean(crps))

    preds = np.array(preds)
    trues = np.array(trues)

    rmse = float(np.sqrt(np.mean((preds - trues)**2)))
    nll = float(np.mean(nlls))
    crps = float(np.mean(crps_list))

    return {
      
        "RMSE": rmse,
        "NLL_mean": nll,
        "CRPS_mean": crps,
    }

