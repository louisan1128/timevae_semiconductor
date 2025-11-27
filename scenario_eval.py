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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def plot_fanchart_long(true_seq_full, 
                       pred_seq_last, 
                       scenario_samples, 
                       feature_index=0, 
                       history=60):
    """
    true_seq_full : 전체 Y 시계열 (N,H,D의 Y 말고, 원래 raw나 scaled 전체)
    pred_seq_last : 마지막 chunk reconstruction (H,D)
    scenario_samples : (num_samples, H, D)
    history : 몇 개의 실제 과거 데이터 보여줄지
    """

    scenario_samples = np.array(scenario_samples)
    lower = np.percentile(scenario_samples[:, :, feature_index], 10, axis=0)
    median = np.percentile(scenario_samples[:, :, feature_index], 50, axis=0)
    upper = np.percentile(scenario_samples[:, :, feature_index], 90, axis=0)

    H = pred_seq_last.shape[0]

    # 최근 history 길이만큼 자르기
    true_recent = true_seq_full[-history:]

    # 길이 맞추기용 x축
    t_history = list(range(len(true_recent)))
    t_future  = list(range(len(true_recent), len(true_recent) + H))

    plt.figure(figsize=(12, 6))

    # 1) History part
    plt.plot(t_history, true_recent[:, feature_index], color="black", label="History (True)")

    # 2) Prediction (one-shot reconstruction)
    plt.plot(t_future, pred_seq_last[:, feature_index], color="blue", label="Prediction")

    # 3) Scenario median
    plt.plot(t_future, median, color="red", label="Median Scenario")

    # 4) Scenario band (10~90%)
    plt.fill_between(t_future, lower, upper, color="red", alpha=0.2)

    # Titles, etc.
    plt.title("Long Horizon Fan Chart (History + Forecast)")
    plt.grid(True)
    plt.legend()
    plt.show()

def rolling_posterior_forecast(
    model_path, X, C,
    latent_dim, cond_dim, hidden, H, beta,
    device="cuda"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dim = X.shape[-1]

    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H).to(device)
    prior = ConditionalPrior(cond_dim, latent_dim, hidden).to(device)

    model = TimeVAE(encoder, decoder, prior, latent_dim, beta).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds = []

    with torch.no_grad():
        for t in range(len(X)):
            x = torch.tensor(X[t:t+1]).float().to(device)
            c = torch.tensor(C[t:t+1]).float().to(device)

            mu_q, logvar_q = model.encoder(x, c)
            z = mu_q
            mean, _ = model.decoder(z, c)

            preds.append(mean.cpu().numpy()[0, 0, :])

    return np.array(preds)  # (N, D)

def posterior_scenario(
    model_path, X_last, C_last,
    latent_dim, cond_dim, hidden, H, beta,
    num_samples=30, shrink=0.1, device="cuda"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dim = X_last.shape[-1]

    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H).to(device)
    prior = ConditionalPrior(cond_dim, latent_dim, hidden).to(device)

    model = TimeVAE(encoder, decoder, prior, latent_dim, beta).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    X_t = torch.tensor(X_last).float().unsqueeze(0).to(device)
    C_t = torch.tensor(C_last).float().unsqueeze(0).to(device)

    with torch.no_grad():
        mu_q, logvar_q = model.encoder(X_t, C_t)
        std = torch.exp(0.5 * logvar_q) * shrink

        samples = []
        for _ in range(num_samples):
            eps = torch.randn_like(std)
            z = mu_q + eps * std
            mean, _ = model.decoder(z, C_t)
            samples.append(mean.cpu().numpy()[0])

    return np.array(samples)
def plot_full_forecast_and_scenario(
    true_full,
    forecast_full,
    scenario_samples,
    feature_index=0,
    H=12
):
    true_full = np.array(true_full)
    forecast_full = np.array(forecast_full)
    scenario_samples = np.array(scenario_samples)

    N = len(true_full)
    t = np.arange(N)

    # Scenario percentiles
    lower = np.percentile(scenario_samples[:, :, feature_index], 10, axis=0)
    median = np.percentile(scenario_samples[:, :, feature_index], 50, axis=0)
    upper = np.percentile(scenario_samples[:, :, feature_index], 90, axis=0)

    t_future = np.arange(N, N + H)

    plt.figure(figsize=(14, 6))

    # 1) True 전체
    plt.plot(t, true_full[:, feature_index], color="black", label="History (True)")

    # 2) Rolling forecast
    plt.plot(t, forecast_full[:, feature_index], color="blue", label="Prediction")

    # 3) Scenario fan chart
    plt.plot(t_future, median, color="red", label="Median Scenario")
    plt.fill_between(t_future, lower, upper, color="red", alpha=0.25)

    plt.title("Full Horizon: True vs Prediction vs Scenario")
    plt.grid(True)
    plt.legend()
    plt.show()
