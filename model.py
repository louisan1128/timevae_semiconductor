# ===========================================
# model.py — Encoder / Decoder / TimeVAE
# ===========================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import StudentT
from torch.nn.utils import weight_norm


# -------------------------------
# Condition Layer
# -------------------------------
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


# -------------------------------
# TCN 기반 Layer들
# -------------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


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


# -------------------------------
# Encoder
# -------------------------------
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


# -------------------------------
# Decoder
# -------------------------------
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


# -------------------------------
# Conditional Prior
# -------------------------------
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


# -------------------------------
# Full CT-VAE (TimeVAE)
# -------------------------------
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
