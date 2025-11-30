# ===========================================
# scenario_eval_utils.py
# (Scenario generation + Evaluation + Plotting)
# ===========================================

import math
import numpy as np
import matplotlib.pyplot as plt
import torch

from model import Encoder, Decoder, ConditionalPrior, TimeVAE
from macro_pretrain import MacroEncoder


# -------------------------------
# Helper: load model (consistent)
# -------------------------------
def _load_timevae(
    *,
    model_path,
    out_dim,
    cond_dim,
    hidden,
    latent_dim,
    H,
    beta,
    macro_feature_indices,
    macro_hidden_dim,
    macro_latent_dim,
    device,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)

    # macro encoder
    macro_input_dim = len(macro_feature_indices)
    macro_encoder = MacroEncoder(
        input_dim=macro_input_dim,
        hidden_dim=macro_hidden_dim,
        latent_dim=macro_latent_dim,
    ).to(device)
    macro_encoder.load_state_dict(torch.load("macro_encoder.pth", map_location=device))
    macro_encoder.eval()
    for p in macro_encoder.parameters():
        p.requires_grad = False

    # timevae
    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H).to(device)
    prior = ConditionalPrior(cond_dim, macro_latent_dim, latent_dim, hidden).to(device)

    model = TimeVAE(
        encoder=encoder,
        decoder=decoder,
        prior=prior,
        macro_encoder=macro_encoder,
        latent_dim=latent_dim,
        beta=beta,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


# -------------------------------
# Evaluation (Posterior Reconstruction)
# -------------------------------
def evaluate_model(
    model_path, X, Y, C,
    latent_dim, cond_dim, hidden, H,
    beta, macro_feature_indices,
    macro_hidden_dim, macro_latent_dim,
    device="cuda",
):
    """
    NOTE: 여기서는 '진짜 posterior mean'으로 평가하려고
    z = mu_q (deterministic) 를 사용한다.
    """
    out_dim = X.shape[-1]
    model, device = _load_timevae(
        model_path=model_path,
        out_dim=out_dim,
        cond_dim=cond_dim,
        hidden=hidden,
        latent_dim=latent_dim,
        H=H,
        beta=beta,
        macro_feature_indices=macro_feature_indices,
        macro_hidden_dim=macro_hidden_dim,
        macro_latent_dim=macro_latent_dim,
        device=device,
    )

    preds, trues = [], []

    with torch.no_grad():
        for i in range(len(X)):
            x = torch.tensor(X[i:i + 1]).float().to(device)   # (1,L,D)
            c = torch.tensor(C[i:i + 1]).float().to(device)   # (1,cond_dim)
            y_true = torch.tensor(Y[i:i + 1]).float().to(device)  # (1,H,D)

            # deterministic posterior mean
            mu_q, logvar_q = model.encoder(x, c)
            mean, dist = model.decoder(mu_q, c)

            preds.append(mean.cpu().numpy())
            trues.append(y_true.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    mse = float(np.mean((preds - trues) ** 2))
    return preds, trues, mse


# -------------------------------
# Scenario Sampling Near Posterior Anchor
# -------------------------------
def scenario_predict_local(
    model_path,
    X_last, cond_true, cond_scenario,
    latent_dim, cond_dim, hidden, H, beta,
    num_samples, z_shrink,
    macro_feature_indices,
    macro_hidden_dim, macro_latent_dim,
    device="cuda",
):
    """
    posterior q(z|x,c_true)를 anchor로 쓰고,
    그 주변에서만 작은 noise를 주어 scenario를 여러 개 샘플링한다.
    (Student-t dist에서 sample)
    """
    out_dim = X_last.shape[-1]
    model, device = _load_timevae(
        model_path=model_path,
        out_dim=out_dim,
        cond_dim=cond_dim,
        hidden=hidden,
        latent_dim=latent_dim,
        H=H,
        beta=beta,
        macro_feature_indices=macro_feature_indices,
        macro_hidden_dim=macro_hidden_dim,
        macro_latent_dim=macro_latent_dim,
        device=device,
    )

    X_t = torch.tensor(X_last[None]).float().to(device)            # (1,L,D)
    C_t = torch.tensor(cond_true[None]).float().to(device)         # (1,cond_dim)
    C_s = torch.tensor(cond_scenario[None]).float().to(device)     # (1,cond_dim)

    with torch.no_grad():
        mu_q, logvar_q = model.encoder(X_t, C_t)
        std_q = torch.exp(0.5 * logvar_q)

    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            eps = torch.randn_like(std_q)
            z = mu_q + z_shrink * std_q * eps

            mean_scen, dist_scen = model.decoder(z, C_s)
            y_scen = dist_scen.sample()  # safer than rsample for StudentT

            samples.append(y_scen.squeeze(0).cpu().numpy())  # (H,D)

    return np.stack(samples, axis=0)  # (num_samples, H, D)


# -------------------------------
# Fan Chart Plotting
# -------------------------------
def plot_fanchart(true_seq, pred_seq, scenario_samples, feature_index=0):
    scenario_samples = np.array(scenario_samples).squeeze()

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


def plot_fanchart_long(true_seq_full, pred_seq_last, scenario_samples, feature_index=0, history=60):
    scenario_samples = np.array(scenario_samples).squeeze()
    lower = np.percentile(scenario_samples[:, :, feature_index], 10, axis=0)
    median = np.percentile(scenario_samples[:, :, feature_index], 50, axis=0)
    upper = np.percentile(scenario_samples[:, :, feature_index], 90, axis=0)

    H = pred_seq_last.shape[0]
    true_recent = true_seq_full[-history:]

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


# -------------------------------
# Rolling posterior forecast (deterministic)
# -------------------------------
def rolling_posterior_forecast(
    model_path, X, C,
    latent_dim, cond_dim, hidden, H, beta,
    macro_feature_indices,
    macro_hidden_dim, macro_latent_dim,
    device="cuda",
):
    """
    deterministic posterior mean forecast:
    mu_q -> decoder(mu_q, c)
    """
    out_dim = X.shape[-1]
    model, device = _load_timevae(
        model_path=model_path,
        out_dim=out_dim,
        cond_dim=cond_dim,
        hidden=hidden,
        latent_dim=latent_dim,
        H=H,
        beta=beta,
        macro_feature_indices=macro_feature_indices,
        macro_hidden_dim=macro_hidden_dim,
        macro_latent_dim=macro_latent_dim,
        device=device,
    )

    preds = []
    with torch.no_grad():
        for t in range(len(X)):
            x = torch.tensor(X[t:t + 1]).float().to(device)
            c = torch.tensor(C[t:t + 1]).float().to(device)

            mu_q, logvar_q = model.encoder(x, c)
            mean, dist = model.decoder(mu_q, c)

            preds.append(mean[0, 0, :].cpu().numpy())  # 1-step only

    return np.array(preds)


# -------------------------------
# Posterior scenario sampling (Student-t samples)
# -------------------------------
def posterior_scenario(
    model_path, X_last, C_last,
    latent_dim, cond_dim, hidden, H, beta,
    macro_feature_indices,
    macro_hidden_dim, macro_latent_dim,
    num_samples=30, shrink=0.2,
    device="cuda",
):
    out_dim = X_last.shape[-1]
    model, device = _load_timevae(
        model_path=model_path,
        out_dim=out_dim,
        cond_dim=cond_dim,
        hidden=hidden,
        latent_dim=latent_dim,
        H=H,
        beta=beta,
        macro_feature_indices=macro_feature_indices,
        macro_hidden_dim=macro_hidden_dim,
        macro_latent_dim=macro_latent_dim,
        device=device,
    )

    X_t = torch.tensor(X_last[None]).float().to(device)  # (1,L,D)
    C_t = torch.tensor(C_last[None]).float().to(device)  # (1,cond_dim)

    with torch.no_grad():
        mu_q, logvar_q = model.encoder(X_t, C_t)
        std_q = torch.exp(0.5 * logvar_q)

    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            eps = torch.randn_like(std_q)
            z = mu_q + shrink * std_q * eps
            mean, dist = model.decoder(z, C_t)
            y = dist.sample()
            samples.append(y.squeeze(0).cpu().numpy())

    return np.stack(samples, axis=0)


def plot_full_forecast_and_scenario(true_full, forecast_full, scenario_samples, feature_index=0, H=12):
    true_full = np.array(true_full)
    forecast_full = np.array(forecast_full)
    scenario_samples = np.array(scenario_samples)

    N = len(true_full)
    t = np.arange(N)

    lower = np.percentile(scenario_samples[:, :, feature_index], 10, axis=0).squeeze()
    median = np.percentile(scenario_samples[:, :, feature_index], 50, axis=0).squeeze()
    upper = np.percentile(scenario_samples[:, :, feature_index], 90, axis=0).squeeze()

    t_future = np.arange(N, N + H)

    plt.figure(figsize=(14, 6))
    plt.plot(t, true_full[:, feature_index], color="black", label="History (True)")
    plt.plot(t, forecast_full[:, feature_index], color="blue", label="Prediction")
    plt.plot(t_future, median, color="red", label="Median Scenario")
    plt.fill_between(t_future, lower, upper, color="red", alpha=0.25)
    plt.title("Full Horizon: True vs Prediction vs Scenario")
    plt.grid(True)
    plt.legend()
    plt.show()


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

    return {"MSE": mse, "MAE": mae, "RMSE": rmse}


def compute_coverage_and_sharpness(scenario_samples, true_future, feature_index=0, lower_q=10, upper_q=90):
    scenario_samples = np.array(scenario_samples)
    true_future = np.array(true_future)

    lower = np.percentile(scenario_samples[:, :, feature_index], lower_q, axis=0)
    upper = np.percentile(scenario_samples[:, :, feature_index], upper_q, axis=0)
    true = true_future[:, feature_index]

    inside = (true >= lower) & (true <= upper)
    coverage = inside.mean()

    width = upper - lower
    sharpness = width.mean()

    return {f"Coverage_{upper_q-lower_q}%": coverage, f"Sharpness_{upper_q-lower_q}%": sharpness}


def compute_crps_from_samples(scenario_samples, true_future, feature_index=0):
    S = np.array(scenario_samples)[:, :, feature_index]  # (M,H)
    S = np.squeeze(S)
    y = np.array(true_future)[:, feature_index]          # (H,)

    term1 = np.mean(np.abs(S - y[None, :]), axis=0)  # (H,)

    S1 = S[:, None, :]  # (M,1,H)
    S2 = S[None, :, :]  # (1,M,H)
    term2 = np.mean(np.abs(S1 - S2), axis=(0, 1))  # (H,)

    crps_per_h = term1 - 0.5 * term2
    crps = crps_per_h.mean()
    return {"CRPS_mean": crps, "CRPS_per_h": crps_per_h}


def evaluate_student_t_nll(
    model_path, X, Y, C,
    latent_dim, cond_dim, hidden, H, beta,
    macro_feature_indices,
    macro_hidden_dim, macro_latent_dim,
    device="cuda",
):
    """
    train과 동일 경로(y!=None)로 recon_loss = -log_prob.mean() 집계
    """
    out_dim = X.shape[-1]
    model, device = _load_timevae(
        model_path=model_path,
        out_dim=out_dim,
        cond_dim=cond_dim,
        hidden=hidden,
        latent_dim=latent_dim,
        H=H,
        beta=beta,
        macro_feature_indices=macro_feature_indices,
        macro_hidden_dim=macro_hidden_dim,
        macro_latent_dim=macro_latent_dim,
        device=device,
    )

    nll_list = []
    with torch.no_grad():
        for i in range(len(X)):
            x = torch.tensor(X[i:i + 1]).float().to(device)
            y = torch.tensor(Y[i:i + 1]).float().to(device)
            c = torch.tensor(C[i:i + 1]).float().to(device)

            macro_x = x[:, :, macro_feature_indices].permute(0, 2, 1)

            loss, recon, kl, mean, z, prior_stats = model(
                x, c, macro_x,
                y=y,
                use_prior_sampling_if_no_y=False
            )
            nll_list.append(float(recon.item()))

    nll_arr = np.array(nll_list, dtype=np.float32)
    return {
        "NLL_mean": float(nll_arr.mean()),
        "NLL_std": float(nll_arr.std()),
        "NLL_per_sample": nll_arr,
    }


def compute_risk_metrics(
    scenario_samples,
    current_level_scaled,
    scaler,
    feature_index=0,
    horizon_idx=-1,
    tail_threshold_raw=-0.1,
    alpha=0.10,
):
    scenario_samples = np.array(scenario_samples)
    future_scaled = scenario_samples[:, horizon_idx, :]  # (M,D)
    future_raw = scaler.inverse_transform(future_scaled)[:, feature_index]

    current_raw = scaler.inverse_transform(np.array(current_level_scaled).reshape(1, -1))[0, feature_index]
    ret = (future_raw - current_raw) / current_raw

    p_up = np.mean(ret > 0)
    p_tail = np.mean(ret < tail_threshold_raw)
    var_alpha = np.quantile(ret, alpha)

    tail = ret[ret <= var_alpha]
    es_alpha = var_alpha if len(tail) < 3 else tail.mean()

    return {
        "P_up": p_up,
        f"P_tail(<{tail_threshold_raw * 100:.1f}%)": p_tail,
        f"VaR_{int(alpha*100)}%": var_alpha,
        f"ES_{int(alpha*100)}%": es_alpha,
    }
