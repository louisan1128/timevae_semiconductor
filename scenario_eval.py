# ===========================================
# scenario_eval.py
# (Scenario generation + Evaluation + Plotting + Metrics)
# ===========================================

import math
import numpy as np
import matplotlib.pyplot as plt
import torch

from model import Encoder, Decoder, ConditionalPrior, TimeVAE
from macro_pretrain import MacroEncoder


# -------------------------------
# Posterior Reconstruction Evaluation
# -------------------------------
def evaluate_model(
    model_path,
    X, Y, C, MACRO_X,
    latent_dim, cond_dim, hidden, H,
    beta,
    macro_hidden_dim, macro_latent_dim,
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

    preds, trues = [], []
    with torch.no_grad():
        for i in range(len(X)):
            x = torch.tensor(X[i:i+1]).float().to(device)
            y = torch.tensor(Y[i:i+1]).float().to(device)
            c = torch.tensor(C[i:i+1]).float().to(device)
            mx = torch.tensor(MACRO_X[i:i+1]).float().to(device)

            loss, recon, kl, mean, _, _ = model(
                x, c, mx,
                y=y,
                use_prior_sampling_if_no_y=False
            )
            preds.append(mean.cpu().numpy())
            trues.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    mse = float(np.mean((preds - trues) ** 2))
    return preds, trues, mse


# -------------------------------
# Scenario Sampling Near Posterior Anchor
# -------------------------------
def scenario_predict_local(
    model_path,
    X_last,
    macro_x_last,
    cond_true,
    cond_scenario,
    latent_dim, cond_dim, hidden, H, beta,
    num_samples,
    z_shrink,
    macro_hidden_dim, macro_latent_dim,
    device="cuda",
    macro_encoder_path="macro_encoder.pth",
    sample_from_student_t=True,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    out_dim = X_last.shape[-1]
    macro_input_dim = macro_x_last.shape[0]  # (macro_dim, L)

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

    X_t = torch.tensor(X_last[None]).float().to(device)            # (1,L,D)
    C_t = torch.tensor(cond_true[None]).float().to(device)         # (1,cond)
    C_s = torch.tensor(cond_scenario[None]).float().to(device)     # (1,cond)
    MX_t = torch.tensor(macro_x_last[None]).float().to(device)     # (1,macro_dim,L)

    with torch.no_grad():
        mu_q, logvar_q = model.encoder(X_t, C_t)
        std_q = torch.exp(0.5 * logvar_q)

    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            eps = torch.randn_like(std_q)
            z = mu_q + z_shrink * std_q * eps
            mean_scen, dist_scen = model.decoder(z, C_s)
            y_scen = dist_scen.rsample() if sample_from_student_t else mean_scen
            samples.append(y_scen.squeeze(0).cpu().numpy())

    return np.stack(samples, axis=0)  # (num_samples, H, D)


# -------------------------------
# Plotting
# -------------------------------
def plot_fanchart(true_seq, pred_seq, scenario_samples, feature_index=0):
    S = np.array(scenario_samples)
    lower = np.percentile(S[:, :, feature_index], 10, axis=0)
    median = np.percentile(S[:, :, feature_index], 50, axis=0)
    upper = np.percentile(S[:, :, feature_index], 90, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(true_seq[:, feature_index], label="True", color="black")
    plt.plot(pred_seq[:, feature_index], label="Prediction", color="blue")
    plt.plot(median, color="red", label="Median Scenario")
    plt.fill_between(range(len(median)), lower, upper, color="red", alpha=0.2)
    plt.title("Fan Chart (Scenario Forecast)")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_fanchart_long(true_seq_full, pred_seq_last, scenario_samples, feature_index=0, history=60):
    S = np.array(scenario_samples)
    lower = np.percentile(S[:, :, feature_index], 10, axis=0)
    median = np.percentile(S[:, :, feature_index], 50, axis=0)
    upper = np.percentile(S[:, :, feature_index], 90, axis=0)

    H = pred_seq_last.shape[0]
    true_recent = np.array(true_seq_full)[-history:]

    t_history = list(range(len(true_recent)))
    t_future = list(range(len(true_recent), len(true_recent) + H))

    plt.figure(figsize=(12, 6))
    plt.plot(t_history, true_recent[:, feature_index], color="black", label="History (True)")
    plt.plot(t_future, pred_seq_last[:, feature_index], color="blue", label="Prediction")
    plt.plot(t_future, median, color="red", label="Median Scenario")
    plt.fill_between(t_future, lower, upper, color="red", alpha=0.2)
    plt.title("Long Horizon Fan Chart (History + Forecast)")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_full_forecast_and_scenario(true_full, forecast_full, scenario_samples, feature_index=0, H=12):
    true_full = np.array(true_full)
    forecast_full = np.array(forecast_full)
    S = np.array(scenario_samples)

    N = len(true_full)
    t = np.arange(N)

    lower = np.percentile(S[:, :, feature_index], 10, axis=0).squeeze()
    median = np.percentile(S[:, :, feature_index], 50, axis=0).squeeze()
    upper = np.percentile(S[:, :, feature_index], 90, axis=0).squeeze()
    t_future = np.arange(N, N + H)

    plt.figure(figsize=(14, 6))
    plt.plot(t, true_full[:, feature_index], color="black", label="History (True)")
    plt.plot(t, forecast_full[:, feature_index], color="blue", label="Prediction (rolling posterior mean)")
    plt.plot(t_future, median, color="red", label="Median Scenario")
    plt.fill_between(t_future, lower, upper, color="red", alpha=0.25)
    plt.title("Full Horizon: True vs Prediction vs Scenario")
    plt.grid(True)
    plt.legend()
    plt.show()


# -------------------------------
# Rolling posterior forecast (1-step from posterior mean)
# -------------------------------
def rolling_posterior_forecast(
    model_path,
    X, C, MACRO_X,
    latent_dim, cond_dim, hidden, H, beta,
    macro_hidden_dim, macro_latent_dim,
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

    preds = []
    with torch.no_grad():
        for t in range(len(X)):
            x = torch.tensor(X[t:t+1]).float().to(device)
            c = torch.tensor(C[t:t+1]).float().to(device)

            mu_q, logvar_q = model.encoder(x, c)
            mean, _ = model.decoder(mu_q, c)
            preds.append(mean[0, 0, :].cpu().numpy())

    return np.array(preds)


# -------------------------------
# Posterior scenario sampling
# -------------------------------
def posterior_scenario(
    model_path,
    X_last, C_last, macro_x_last,
    latent_dim, cond_dim, hidden, H, beta,
    macro_hidden_dim, macro_latent_dim,
    num_samples=30,
    shrink=0.2,
    device="cuda",
    macro_encoder_path="macro_encoder.pth",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    out_dim = X_last.shape[-1]
    macro_input_dim = macro_x_last.shape[0]

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

    X_t = torch.tensor(X_last[None]).float().to(device)
    C_t = torch.tensor(C_last[None]).float().to(device)

    with torch.no_grad():
        mu_q, logvar_q = model.encoder(X_t, C_t)
        std_q = torch.exp(0.5 * logvar_q)

    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            eps = torch.randn_like(std_q)
            z = mu_q + shrink * std_q * eps
            mean, _ = model.decoder(z, C_t)
            samples.append(mean.squeeze(0).cpu().numpy())

    return np.stack(samples, axis=0)


# ===============================
# Metrics
# ===============================
def compute_point_forecast_metrics(preds, trues):
    preds = np.array(preds)
    trues = np.array(trues)
    assert preds.shape == trues.shape, "preds와 trues shape가 다름"
    diff = preds - trues
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(mse)
    return {"MSE": float(mse), "MAE": float(mae), "RMSE": float(rmse)}


def compute_coverage_and_sharpness(scenario_samples, true_future, feature_index=0, lower_q=10, upper_q=90):
    S = np.array(scenario_samples)
    y = np.array(true_future)
    lower = np.percentile(S[:, :, feature_index], lower_q, axis=0)
    upper = np.percentile(S[:, :, feature_index], upper_q, axis=0)
    truth = y[:, feature_index]
    inside = (truth >= lower) & (truth <= upper)
    coverage = inside.mean()
    sharpness = (upper - lower).mean()
    return {
        f"Coverage_{upper_q-lower_q}%": float(coverage),
        f"Sharpness_{upper_q-lower_q}%": float(sharpness),
    }


def compute_crps_from_samples(scenario_samples, true_future, feature_index=0):
    S = np.array(scenario_samples)[:, :, feature_index]  # (M,H)
    y = np.array(true_future)[:, feature_index]          # (H,)
    M, H = S.shape

    term1 = np.mean(np.abs(S - y[None, :]), axis=0)
    S1 = S[:, None, :]
    S2 = S[None, :, :]
    term2 = np.mean(np.abs(S1 - S2), axis=(0, 1))

    crps_per_h = term1 - 0.5 * term2
    crps = crps_per_h.mean()
    return {"CRPS
