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



















########################
####evaluation#########
#########################

# ===============================
# Forecast Metrics (MSE / MAE / RMSE)
# ===============================

def compute_point_forecast_metrics(preds, trues):
    """
    preds: (N, D) or (N, H, D) 모두 가능. trues와 동일 shape 가정.
    trues: same shape as preds
    return: dict { 'MSE': ..., 'MAE': ..., 'RMSE': ... }
    """
    preds = np.array(preds)
    trues = np.array(trues)

    assert preds.shape == trues.shape, "preds와 trues shape가 다름"

    diff = preds - trues
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(mse)

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
    }

# ===============================
# Probabilistic Metrics (CRPS, Coverage, Sharpness)
# ===============================

def compute_coverage_and_sharpness(
    scenario_samples,
    true_future,
    feature_index=0,
    lower_q=10,
    upper_q=90
):
    """
    scenario_samples: (num_samples, H, D)
    true_future:      (H, D)  -> 마지막 H-step의 실제 Y
    feature_index:    어느 feature를 평가할지
    lower_q, upper_q: 예: 10,90 => 80% 구간
    """
    scenario_samples = np.array(scenario_samples)
    true_future = np.array(true_future)

    lower = np.percentile(scenario_samples[:, :, feature_index], lower_q, axis=0)
    upper = np.percentile(scenario_samples[:, :, feature_index], upper_q, axis=0)
    true = true_future[:, feature_index]

    inside = (true >= lower) & (true <= upper)
    coverage = inside.mean()

    # Sharpness: 구간 폭의 평균
    width = upper - lower
    sharpness = width.mean()

    return {
        f"Coverage_{upper_q-lower_q}%": coverage,
        f"Sharpness_{upper_q-lower_q}%": sharpness
    }

def compute_crps_from_samples(scenario_samples, true_future, feature_index=0):
    """
    CRPS ~ E|S - y| - 0.5 E|S - S'|
    scenario_samples: (num_samples, H, D)
    true_future:      (H, D)
    """
    S = np.array(scenario_samples)[:, :, feature_index]  # (M, H)
    y = np.array(true_future)[:, feature_index]          # (H,)

    M, H = S.shape
    # E|S - y|
    term1 = np.mean(np.abs(S - y[None, :]), axis=0)  # (H,)

    # E|S - S'|
    # (M, H) vs (M, H) broadcast -> (M, M, H) 이라 메모리 크면 위험해서
    # 조금 단순화: 일부 샘플만 사용하거나, 행 샘플링
    # 여기서는 M이 크지 않다고 가정하고 full 사용
    S1 = S[:, None, :]  # (M,1,H)
    S2 = S[None, :, :]  # (1,M,H)
    term2 = np.mean(np.abs(S1 - S2), axis=(0,1))  # (H,)

    crps_per_h = term1 - 0.5 * term2
    crps = crps_per_h.mean()

    return {
        "CRPS_mean": crps,
        "CRPS_per_h": crps_per_h
    }



# ===============================
# Student-t NLL (Decoder와 맞춘 버전)
# ===============================

def student_t_nll_torch(y, mean, scale, df, eps=1e-6):
    """
    y, mean, scale, df: torch.Tensor (broadcast 가능), shape 대략 (B, H, D)
    TimeVAE decoder가 내놓는 Student-t likelihood와 맞추기 위한 NLL.
    """
    # 안정성용 epsilon
    scale = scale + eps
    df = df + eps

    # (y - μ) / σ
    t = (y - mean) / scale

    # log 정규화 항
    # log Γ((ν+1)/2) - log Γ(ν/2) - 0.5 log(νπ) - log σ
    log_norm = (
        torch.lgamma((df + 1.0) / 2.0)
        - torch.lgamma(df / 2.0)
        - 0.5 * torch.log(df * math.pi)
        - torch.log(scale)
    )

    # kernel 부분: - (ν+1)/2 * log(1 + t^2 / ν)
    log_kernel = - (df + 1.0) / 2.0 * torch.log1p((t ** 2) / df)

    log_pdf = log_norm + log_kernel
    nll = -log_pdf  # Negative log-likelihood

    return nll  # shape 그대로 (B, H, D)


def evaluate_student_t_nll(
    model_path, X, Y, C,
    latent_dim, cond_dim, hidden, H, beta,
    device="cuda"
):
    """
    Decoder가 return mean, dist 형태일 때 맞는 Student-t NLL 평가 함수.
    dist는 torch.distributions.StudentT 객체이어야 한다.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dim = X.shape[-1]

    # 모델 로드
    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H).to(device)
    prior   = ConditionalPrior(cond_dim, latent_dim, hidden).to(device)

    model = TimeVAE(encoder, decoder, prior, latent_dim, beta).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    nll_list = []

    with torch.no_grad():
        for i in range(len(X)):

            x = torch.tensor(X[i:i+1]).float().to(device)     # (1, L, D)
            y = torch.tensor(Y[i:i+1]).float().to(device)     # (1, H, D)
            c = torch.tensor(C[i:i+1]).float().to(device)     # (1, cond_dim)

            # posterior mean 사용
            mu_q, logvar_q = model.encoder(x, c)
            z = mu_q

            # 디코더: mean, dist 반환
            mean, dist = model.decoder(z, c)

            # dist는 torch.distributions.StudentT 객체
            # → dist.log_prob(y)는 (1, H, D)
            log_pdf = dist.log_prob(y)

            # Negative Log Likelihood = -log p(y|z,c)
            nll = -log_pdf

            # sample 하나에 대해 모든 H,D 평균
            nll_mean = nll.mean().item()
            nll_list.append(nll_mean)

    nll_arr = np.array(nll_list)

    return {
        "NLL_mean": float(nll_arr.mean()),
        "NLL_std": float(nll_arr.std()),
        "NLL_per_sample": nll_arr,
    }



# ===============================
# Scenario-based Risk Metrics
# ===============================

def compute_risk_metrics(
    scenario_samples,
    current_level,
    feature_index=0,
    horizon_idx=-1,
    tail_threshold=-0.1,  # -10% 이하 하락
    alpha=0.10            # 10% VaR, ES
):
    """
    scenario_samples: (num_samples, H, D)
    current_level: scalar (현재 Export 수준) 또는 같은 스케일의 값
    feature_index: 평가할 feature index
    horizon_idx: 어느 horizon을 볼지 (기본 -1: 마지막)
    tail_threshold: e.g., -0.1 => -10% 이하 하락
    alpha: VaR, ES의 신뢰수준(0.1 => 10%)
    """
    scenario_samples = np.array(scenario_samples)  # (M,H,D)
    M, H, D = scenario_samples.shape

    future = scenario_samples[:, horizon_idx, feature_index]  # (M,)

    # 수익률 or 변화율: (future - current) / current
    # (데이터 스케일에 따라 조정 가능)
    ret = (future - current_level) / current_level

    # 상승 확률
    p_up = np.mean(ret > 0)

    # Tail Risk (P(ret < -x%))
    p_tail = np.mean(ret < tail_threshold)

    # VaR_α, ES_α (하단 tail)
    var_alpha = np.quantile(ret, alpha)
    es_alpha = ret[ret <= var_alpha].mean() if np.any(ret <= var_alpha) else var_alpha

    return {
        "P_up": p_up,
        f"P_tail(<{tail_threshold*100:.1f}%)": p_tail,
        f"VaR_{int(alpha*100)}%": var_alpha,
        f"ES_{int(alpha*100)}%": es_alpha,
    }



# ===============================
# Ablation Utilities
# ===============================

def compare_models_point_forecast(
    model_paths,
    X, Y, C,
    latent_dim, cond_dim, hidden, H, beta,
    device="cuda"
):
    """
    model_paths: dict, 예) {
        "StudentT_withPrior_FiLM": "ckpt/st_t_prior_film.pth",
        "Gaussian_noPrior_PlainTCN": "ckpt/gauss_no_prior_plain.pth",
        ...
    }
    각 모델에 대해 MSE/MAE/RMSE 계산해서 dict로 반환
    """
    results = {}

    for name, path in model_paths.items():
        preds, trues, _ = evaluate_model(
            path, X, Y, C,
            latent_dim, cond_dim, hidden, H, beta,
            device=device
        )
        metrics = compute_point_forecast_metrics(preds, trues)
        results[name] = metrics

    return results


def compare_models_probabilistic_nll(
    model_paths,
    X, Y, C,
    latent_dim, cond_dim, hidden, H, beta,
    device="cuda",
    distribution="student_t"
):
    results = {}

    for name, path in model_paths.items():
        metrics = evaluate_student_t_nll(
            path, X, Y, C,
            latent_dim, cond_dim, hidden, H, beta,
            device=device,
            distribution=distribution
        )
        results[name] = metrics

    return results
