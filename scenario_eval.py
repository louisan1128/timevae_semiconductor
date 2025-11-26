# ===========================================
# scenario_eval_utils.py
# (Scenario generation + Evaluation + Plotting)
# ===========================================

import torch
import numpy as np
import matplotlib.pyplot as plt

from model import Encoder, Decoder, ConditionalPrior, TimeVAE


# -------------------------------
# Evaluation (Posterior Reconstruction)
# -------------------------------
def evaluate_model(model_path, X, Y, C, latent_dim, cond_dim, hidden, H, beta, device):
    device = torch.device(device)
    out_dim = X.shape[-1]

    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H).to(device)
    prior = ConditionalPrior(cond_dim, latent_dim, hidden).to(device)

    model = TimeVAE(encoder, decoder, prior, latent_dim, beta).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds, trues = [], []

    with torch.no_grad():
        for i in range(len(X)):
            x = torch.tensor(X[i:i+1]).float().to(device)
            y = torch.tensor(Y[i:i+1]).float().to(device)
            c = torch.tensor(C[i:i+1]).float().to(device)

            # 학습 모드와 동일하게 posterior를 써서 reconstruction
            loss, recon, kl, mean, _ = model(x, c, y)
            preds.append(mean.cpu().numpy())
            trues.append(y.cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)


    mse = np.mean((preds - trues)**2)
    return preds, trues, mse


# -------------------------------
# Scenario Sampling Near Posterior Anchor
# -------------------------------
def scenario_predict_local(
    model_path, X_last, cond_true, cond_scenario,
    latent_dim, cond_dim, hidden, H, beta,
    num_samples=30, z_shrink=0.1, device="cuda"
):
    """
    posterior q(z|x,c_true)를 anchor로 쓰고,
    그 주변에서만 작은 noise를 주어 scenario를 여러 개 샘플링한다.
    """

    device = torch.device(device)
    out_dim = X_last.shape[-1]

    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H).to(device)
    prior = ConditionalPrior(cond_dim, latent_dim, hidden).to(device)

    model = TimeVAE(encoder, decoder, prior, latent_dim, beta).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    X_t = torch.tensor(X_last).float().unsqueeze(0).to(device)
    C_true = torch.tensor(cond_true).float().unsqueeze(0).to(device)
    C_scen = torch.tensor(cond_scenario).float().unsqueeze(0).to(device)

    with torch.no_grad():
        # 1) posterior q(z | x_last, c_true) 계산
        mu_q, logvar_q = model.encoder(X_t, C_true)# (1, latent_dim)

        # 2) posterior mean을 anchor로 사용
        z_base = mu_q # (1, latent_dim)

        # 3) z 분산을 줄여서 truth 근처에서만 샘플
        std_q = torch.exp(0.5 * logvar_q) * z_shrink

        samples = []
        for _ in range(num_samples):
            eps = torch.randn_like(std_q)
            z = z_base + eps * std_q

            mean, _ = model.decoder(z, C_scen) # (1,H,D)
            samples.append(mean.cpu().numpy()[0])

    return np.array(samples)


# -------------------------------
# Fan Chart Plotting
# -------------------------------
def plot_fanchart(true_seq, pred_seq, scenario_samples, feature_index=0):
    scenario_samples = np.array(scenario_samples)
    lower = np.percentile(scenario_samples[:, :, feature_index], 10, axis=0)
    median = np.percentile(scenario_samples[:, :, feature_index], 50, axis=0)
    upper = np.percentile(scenario_samples[:, :, feature_index], 90, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(true_seq[:, feature_index], label="True", color="black")
    plt.plot(pred_seq[:, feature_index], label="Prediction", color="blue")
    plt.plot(median, color="red", label="Median Scenario")
    plt.fill_between(range(len(median)), lower, upper, color="red", alpha=0.2)

    plt.title("Fan Chart (Scenario Forecast)")
    plt.grid(True)
    plt.legend()
    plt.show()
