# ===========================================
# train_ablation_models.py
# CT-VAE Ablation Training (5 models)
# - full_ct_vae
# - no_student_t
# - no_macro_prior
# - no_film
# - no_decomp
# ===========================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import Encoder, Decoder, ConditionalPrior, TimeVAE
from macro_pretrain import MacroEncoder
from data_train import preprocess


# -------------------------------
# Ablation Variant Modules
# -------------------------------

class EncoderNoFiLM(Encoder):
    """
    FiLM + TCN 대신, input_proj + 평균 pooling만 사용하는 간단 Encoder
    (아키텍처는 달라져도, 입력/출력 shape는 동일하게 유지)
    """
    def __init__(self, x_dim, c_dim, h_dim, z_dim):
        super().__init__(x_dim, c_dim, h_dim, z_dim)
        # FiLM_TCN을 안 쓸 것이므로 덮어쓰기
        self.tcn_film = None

    def forward(self, x, c):
        # x: (B, L, x_dim), c: (B, c_dim)
        x = x.permute(0, 2, 1)        # (B, x_dim, L)
        h = self.input_proj(x)        # (B, hidden, L)

        # cond embedding 써도 되고/안 써도 되지만, 간단히 mean pooling만 적용
        h_mean = h.mean(dim=-1)       # (B, hidden)

        mu = self.mu_layer(h_mean)
        logvar = self.logvar_layer(h_mean)
        return mu, logvar


class DecoderGaussian(Decoder):
    """
    Student-t 대신 Gaussian likelihood 사용.
    나머지 구조는 동일.
    """
    def forward(self, z, c):
        mean, dist_student = super().forward(z, c)
        scale = dist_student.scale
        dist = torch.distributions.Normal(loc=mean, scale=scale)
        return mean, dist


class DecoderNoDecomp(Decoder):
    """
    Trend + Seasonality 제거, Residual RNN만 사용하는 Decoder.
    (out_dim, H, hidden 등 shape는 원본과 동일하게 유지)
    """
    def forward(self, z, c):
        device = z.device
        B = z.size(0)
        D = self.out_dim

        # context
        x = torch.cat([z, c], dim=-1)          # (B, latent+cond)
        h = self.fc_context(x)                 # (B, hidden)

        # Residual RNN만 사용
        rnn_input = h.unsqueeze(1).repeat(1, self.H, 1)  # (B,H,hidden)
        rnn_out, _ = self.rnn(rnn_input)                # (B,H,hidden)
        mean = self.rnn_out(rnn_out)                    # (B,H,D)

        # scale / df 모수는 원본 구조 그대로 사용
        scale = nn.functional.softplus(self.fc_scale(h)).view(B, self.H, D) + 1e-4
        df    = nn.functional.softplus(self.fc_df(h)).view(B, self.H, D) + 2.0

        dist = torch.distributions.StudentT(df, loc=mean, scale=scale)
        return mean, dist


class PriorNoMacro(nn.Module):
    """
    macro z 없이, condition c만 사용하는 prior p(z|c).
    """
    def __init__(self, cond_dim, latent_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden, latent_dim)
        self.logvar_head = nn.Linear(hidden, latent_dim)

    def forward(self, c):
        h = self.net(c)
        return self.mu_head(h), self.logvar_head(h)


class TimeVAENoMacroPrior(TimeVAE):
    """
    macro_encoder 및 z_macro 없이 prior p(z|c)만 사용하는 TimeVAE.
    model.py의 TimeVAE를 상속하지만 forward를 override.
    """
    def __init__(self, encoder, decoder, prior, latent_dim, beta=1.0):
        # macro_encoder=None으로 고정
        super().__init__(encoder, decoder, prior, macro_encoder=None, latent_dim=latent_dim, beta=beta)

    def forward(self, x, c, macro_x=None, y=None, use_prior_sampling_if_no_y=True):
        # 1) posterior q(z|x,c)
        mu_q, logvar_q = self.encoder(x, c)

        # 2) prior p(z|c)
        mu_p, logvar_p = self.prior(c)

        # 3) inference 모드
        if (y is None) and use_prior_sampling_if_no_y:
            z = self.reparameterize(mu_p, logvar_p)
            mean, dist = self.decoder(z, c)
            return mean, z, (mu_p, logvar_p)

        # 4) 학습 모드: posterior 사용
        z = self.reparameterize(mu_q, logvar_q)
        mean, dist = self.decoder(z, c)

        log_prob = dist.log_prob(y)
        recon = -log_prob.mean()

        kl = self.kl_gaussian(mu_q, logvar_q, mu_p, logvar_p)
        loss = recon + self.beta * kl

        return loss, recon, kl, mean, z, (mu_p, logvar_p)


# -------------------------------
# Model Builders (train/eval 공통)
# -------------------------------

def build_full(out_dim, cond_dim, latent_dim, hidden, H, macro_latent_dim, device):
    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H).to(device)
    prior   = ConditionalPrior(cond_dim, macro_latent_dim, latent_dim, hidden).to(device)

    macro_encoder = MacroEncoder(
        input_dim=6,
        hidden_dim=128,
        latent_dim=macro_latent_dim
    ).to(device)
    macro_encoder.load_state_dict(torch.load("macro_encoder.pth", map_location=device))
    macro_encoder.eval()
    for p in macro_encoder.parameters():
        p.requires_grad = False

    model = TimeVAE(
        encoder=encoder,
        decoder=decoder,
        prior=prior,
        macro_encoder=macro_encoder,
        latent_dim=latent_dim,
        beta=1.0
    ).to(device)
    return model


def build_no_student_t(out_dim, cond_dim, latent_dim, hidden, H, macro_latent_dim, device):
    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = DecoderGaussian(latent_dim, cond_dim, out_dim, hidden, H).to(device)
    prior   = ConditionalPrior(cond_dim, macro_latent_dim, latent_dim, hidden).to(device)

    macro_encoder = MacroEncoder(
        input_dim=6,
        hidden_dim=128,
        latent_dim=macro_latent_dim
    ).to(device)
    macro_encoder.load_state_dict(torch.load("macro_encoder.pth", map_location=device))
    macro_encoder.eval()
    for p in macro_encoder.parameters():
        p.requires_grad = False

    model = TimeVAE(
        encoder=encoder,
        decoder=decoder,
        prior=prior,
        macro_encoder=macro_encoder,
        latent_dim=latent_dim,
        beta=1.0
    ).to(device)
    return model


def build_no_film(out_dim, cond_dim, latent_dim, hidden, H, macro_latent_dim, device):
    encoder = EncoderNoFiLM(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H).to(device)
    prior   = ConditionalPrior(cond_dim, macro_latent_dim, latent_dim, hidden).to(device)

    macro_encoder = MacroEncoder(
        input_dim=6,
        hidden_dim=128,
        latent_dim=macro_latent_dim
    ).to(device)
    macro_encoder.load_state_dict(torch.load("macro_encoder.pth", map_location=device))
    macro_encoder.eval()
    for p in macro_encoder.parameters():
        p.requires_grad = False

    model = TimeVAE(
        encoder=encoder,
        decoder=decoder,
        prior=prior,
        macro_encoder=macro_encoder,
        latent_dim=latent_dim,
        beta=1.0
    ).to(device)
    return model


def build_no_decomp(out_dim, cond_dim, latent_dim, hidden, H, macro_latent_dim, device):
    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = DecoderNoDecomp(latent_dim, cond_dim, out_dim, hidden, H).to(device)
    prior   = ConditionalPrior(cond_dim, macro_latent_dim, latent_dim, hidden).to(device)

    macro_encoder = MacroEncoder(
        input_dim=6,
        hidden_dim=128,
        latent_dim=macro_latent_dim
    ).to(device)
    macro_encoder.load_state_dict(torch.load("macro_encoder.pth", map_location=device))
    macro_encoder.eval()
    for p in macro_encoder.parameters():
        p.requires_grad = False

    model = TimeVAE(
        encoder=encoder,
        decoder=decoder,
        prior=prior,
        macro_encoder=macro_encoder,
        latent_dim=latent_dim,
        beta=1.0
    ).to(device)
    return model


def build_no_macro_prior(out_dim, cond_dim, latent_dim, hidden, H, macro_latent_dim, device):
    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H).to(device)
    prior   = PriorNoMacro(cond_dim, latent_dim, hidden).to(device)

    model = TimeVAENoMacroPrior(
        encoder=encoder,
        decoder=decoder,
        prior=prior,
        latent_dim=latent_dim,
        beta=1.0
    ).to(device)
    return model


# -------------------------------
# Training Loop
# -------------------------------

def train_single_model(model, X, Y, C, macro_feature_indices, device, epochs=100, batch_size=32, lr=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    N = len(X)
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for i in range(0, N, batch_size):
            x_batch = torch.tensor(X[i:i+batch_size]).float().to(device)
            y_batch = torch.tensor(Y[i:i+batch_size]).float().to(device)
            c_batch = torch.tensor(C[i:i+batch_size]).float().to(device)

            if model.macro_encoder is not None:
                macro_x_batch = x_batch[:, :, macro_feature_indices].permute(0, 2, 1)
            else:
                macro_x_batch = None

            loss, recon, kl, _, _, _ = model(
                x_batch,
                c_batch,
                macro_x_batch,
                y=y_batch,
                use_prior_sampling_if_no_y=False
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {ep:03d}] Loss = {total_loss:.4f}")

    return model


# -------------------------------
# Main
# -------------------------------

def run_all_ablation():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyperparameters (기존 프로젝트와 동일하게 맞춰야 함)
    L = 36
    H = 12
    LATENT_DIM = 32
    HIDDEN = 128
    MACRO_HIDDEN_DIM = 128
    MACRO_LATENT_DIM = 32

    condition_raw_cols = ["Exchange Rate", "CAPEX", "PMI", "CLI", "ISM"]
    macro_cols = ["PMI", "GS10", "M2SL", "UNRATE", "CPIAUCSL", "INDPRO"]

    X, Y, C, scaler, df_raw, df_scaled, macro_feature_indices = preprocess(
        csv_path="data.csv",
        macro_csv_path="macro.csv",
        condition_raw_cols=condition_raw_cols,
        macro_cols=macro_cols,
        L=L,
        H=H
    )

    out_dim = X.shape[-1]
    cond_dim = C.shape[-1]

    configs = {
        "full_ct_vae":      build_full,
        "no_student_t":     build_no_student_t,
        "no_macro_prior":   build_no_macro_prior,
        "no_film":          build_no_film,
        "no_decomp":        build_no_decomp,
    }

    for name, builder in configs.items():
        print("\n====================================")
        print(f"Training Ablation Model: {name}")
        print("====================================")

        model = builder(
            out_dim=out_dim,
            cond_dim=cond_dim,
            latent_dim=LATENT_DIM,
            hidden=HIDDEN,
            H=H,
            macro_latent_dim=MACRO_LATENT_DIM,
            device=device
        )

        model = train_single_model(
            model,
            X, Y, C,
            macro_feature_indices=macro_feature_indices,
            device=device,
            epochs=100,          # 필요하면 늘려
            batch_size=32,
            lr=1e-3
        )

        save_path = f"{name}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    run_all_ablation()
