# ===========================================
# scenario_eval_utils.py
# (Scenario generation + Evaluation + Plotting)
# ===========================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import math

from model import Encoder, Decoder, ConditionalPrior, TimeVAE
from macro_pretrain import MacroEncoder


# -------------------------------
# Evaluation (Posterior Reconstruction)
# -------------------------------
def evaluate_model(
    model_path, X, Y, C,
    latent_dim, cond_dim, hidden, H,
    beta, macro_feature_indices,
    macro_hidden_dim, macro_latent_dim,
    device="cuda"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dim = X.shape[-1]
    # -------------------------
    # 1) macro encoder
    # -------------------------
    macro_input_dim = len(macro_feature_indices)
    macro_encoder = MacroEncoder(
        input_dim=macro_input_dim,
        hidden_dim=macro_hidden_dim,
        latent_dim=macro_latent_dim
    ).to(device)

    macro_encoder.load_state_dict(torch.load("macro_encoder.pth", map_location=device))
    macro_encoder.eval()
    for p in macro_encoder.parameters():
        p.requires_grad = False

    # -------------------------
    # 2) TimeVAE 구조 (train 과 동일)
    # -------------------------
    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H).to(device)

    prior = ConditionalPrior(
        cond_dim=cond_dim,
        macro_latent_dim=macro_latent_dim,
        latent_dim=latent_dim,
        hidden=hidden
    ).to(device)

    model = TimeVAE(
        encoder=encoder,
        decoder=decoder,
        prior=prior,
        macro_encoder=macro_encoder,
        latent_dim=latent_dim,
        beta=beta
    ).to(device)

    # -------------------------
    # 3) Load trained weights
    # -------------------------
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds, trues = [], []

    with torch.no_grad():
        for i in range(len(X)):
            x = torch.tensor(X[i:i+1]).float().to(device)
            c = torch.tensor(C[i:i+1]).float().to(device)
            y_true = torch.tensor(Y[i:i+1]).float().to(device)

            macro_x = x[:, :, macro_feature_indices].permute(0, 2, 1)

            # (posterior mean 사용)
            loss, recon, kl, mean, _, _ = model(
                x, c, macro_x,
                y=y_true,
                use_prior_sampling_if_no_y=False
            )

            # 학습 모드와 동일하게 posterior를 써서 reconstruction
            preds.append(mean.cpu().numpy())
            trues.append(y_true.cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)


    mse = np.mean((preds - trues)**2)
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
    device="cuda"
):
    """
    posterior q(z|x,c_true)를 anchor로 쓰고,
    그 주변에서만 작은 noise를 주어 scenario를 여러 개 샘플링한다.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dim = X_last.shape[-1]

     # -------------------------
    # 1) macro encoder
    # -------------------------
    macro_input_dim = len(macro_feature_indices)

    macro_encoder = MacroEncoder(
        input_dim=macro_input_dim,
        hidden_dim=macro_hidden_dim,
        latent_dim=macro_latent_dim
    ).to(device)

    macro_encoder.load_state_dict(torch.load("macro_encoder.pth", map_location=device))
    macro_encoder.eval()
    for p in macro_encoder.parameters():
        p.requires_grad = False

    # -------------------------
    # 2) TimeVAE 구성 (train과 동일)
    # -------------------------
    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H).to(device)

    prior = ConditionalPrior(cond_dim, macro_latent_dim, latent_dim, hidden).to(device)

    model = TimeVAE(encoder, decoder, prior, macro_encoder, latent_dim, beta).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # -------------------------
    # 4) Prepare inputs
    # -------------------------
    X_t = torch.tensor(X_last[None]).float().to(device)      # (1,L,D)
    C_t = torch.tensor(cond_true[None]).float().to(device)   # (1,cond_dim)
    C_s = torch.tensor(cond_scenario[None]).float().to(device)
    macro_x = X_t[:, :, macro_feature_indices].permute(0,2,1)

    with torch.no_grad():
        mu_q, logvar_q = model.encoder(X_t, C_t)
        std_q = torch.exp(0.5 * logvar_q)

    samples = []

    with torch.no_grad():
        for _ in range(num_samples):
            eps = torch.randn_like(std_q)
            z = mu_q + z_shrink * std_q * eps   # posterior 주변

            mean_scen, dist_scen = model.decoder(z, C_s)
            # ★ 여기서 mean 대신 Student-t에서 샘플
            y_scen = dist_scen.rsample()        # (1,H,D)

            samples.append(y_scen.squeeze(0).cpu().numpy())   # (H,D)

    return np.stack(samples, axis=0)   # (num_samples, H, D)


# -------------------------------
# Fan Chart Plotting
# -------------------------------
def plot_fanchart(true_seq, pred_seq, scenario_samples, feature_index=0):
    scenario_samples = np.array(scenario_samples)
    scenario_samples = scenario_samples.squeeze()  

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

    scenario_samples = np.array(scenario_samples).squeeze()
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
    macro_feature_indices,
    macro_hidden_dim, macro_latent_dim,
    device="cuda"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dim = X.shape[-1]

    macro_input_dim = len(macro_feature_indices)
    macro_encoder = MacroEncoder(
        input_dim=macro_input_dim,
        hidden_dim=macro_hidden_dim,
        latent_dim=macro_latent_dim
    ).to(device)
    macro_encoder.load_state_dict(torch.load("macro_encoder.pth", map_location=device))
    macro_encoder.eval()

    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H).to(device)
    prior   = ConditionalPrior(cond_dim, macro_latent_dim, latent_dim, hidden).to(device)

    model = TimeVAE(encoder, decoder, prior, macro_encoder, latent_dim, beta).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds = []

    with torch.no_grad():
        for t in range(len(X)):
            x = torch.tensor(X[t:t+1]).float().to(device)
            c = torch.tensor(C[t:t+1]).float().to(device)

            macro_x = x[:, :, macro_feature_indices].permute(0,2,1)

            # posterior mean
            mu_q, logvar_q = model.encoder(x, c)
            mean, _ = model.decoder(mu_q, c)

            preds.append(mean[0,0,:].cpu().numpy())

    return np.array(preds)




def posterior_scenario(
    model_path, X_last, C_last,
    latent_dim, cond_dim, hidden, H, beta,
    macro_feature_indices,
    macro_hidden_dim, macro_latent_dim,
    num_samples=30, shrink=0.2,
    device="cuda"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dim = X_last.shape[-1]

    macro_input_dim = len(macro_feature_indices)
    macro_encoder = MacroEncoder(
        input_dim=macro_input_dim,
        hidden_dim=macro_hidden_dim,
        latent_dim=macro_latent_dim
    ).to(device)
    macro_encoder.load_state_dict(torch.load("macro_encoder.pth", map_location=device))
    macro_encoder.eval()
    for p in macro_encoder.parameters():
        p.requires_grad = False

    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H).to(device)
    prior   = ConditionalPrior(cond_dim, macro_latent_dim, latent_dim, hidden).to(device)

    model = TimeVAE(encoder, decoder, prior, macro_encoder, latent_dim, beta).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    X_t = torch.tensor(X_last[None]).float().to(device)
    C_t = torch.tensor(C_last[None]).float().to(device)
    macro_x = X_t[:, :, macro_feature_indices].permute(0,2,1)

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
    lower = np.percentile(scenario_samples[:, :, feature_index], 10, axis=0).squeeze()
    median = np.percentile(scenario_samples[:, :, feature_index], 50, axis=0).squeeze()
    upper = np.percentile(scenario_samples[:, :, feature_index], 90, axis=0).squeeze()

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
    S = np.squeeze(S)
    if S.ndim > 2:
        S = S.reshape(S.shape[0], -1)

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

    device = torch.device(device)
    out_dim = X.shape[-1]

    # ===== model load =====
    macro_encoder = MacroEncoder(6, 128, 32).to(device)
    macro_encoder.load_state_dict(torch.load("macro_encoder.pth"))
    macro_encoder.eval()

    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H).to(device)
    prior   = ConditionalPrior(cond_dim, 32, latent_dim, hidden).to(device)

    model = TimeVAE(encoder, decoder, prior, macro_encoder, latent_dim, beta).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # =======================
    nll_list = []

    with torch.no_grad():
        for i in range(len(X)):

            x = torch.tensor(X[i:i+1]).float().to(device)
            y = torch.tensor(Y[i:i+1]).float().to(device)
            c = torch.tensor(C[i:i+1]).float().to(device)

            # macro latent
            macro_x = x[:, :, :6].permute(0, 2, 1)
            z_macro = macro_encoder(macro_x)

            # posterior mean
            mu_q, logvar_q = model.encoder(x, c)
            z = mu_q

            # decoder already produces StudentT
            mean, dist = model.decoder(z, c)

            log_pdf = dist.log_prob(y)
            nll = -log_pdf.mean().item()

            nll_list.append(nll)

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
    current_level_scaled,
    scaler,
    feature_index=0,
    horizon_idx=-1,
    tail_threshold_raw=-0.1,  # -10% in raw %
    alpha=0.10
):
    scenario_samples = np.array(scenario_samples)
    M, H, D = scenario_samples.shape

    # ----- 1) scaled → raw 복구 -----
    future_scaled = scenario_samples[:, horizon_idx, :]  # (M, D)
    future_raw = scaler.inverse_transform(future_scaled)[:, feature_index]

    current_raw = scaler.inverse_transform(
        np.array(current_level_scaled).reshape(1, -1)
    )[0, feature_index]

    # ----- 2) returns -----
    ret = (future_raw - current_raw) / current_raw

    # ---- 3) metrics ----
    p_up = np.mean(ret > 0)
    p_tail = np.mean(ret < tail_threshold_raw)

    # VaR
    var_alpha = np.quantile(ret, alpha)

    # ES (robust version)
    tail = ret[ret <= var_alpha]
    if len(tail) < 3:
        es_alpha = var_alpha
    else:
        es_alpha = tail.mean()

    return {
        "P_up": p_up,
        f"P_tail(<{tail_threshold_raw*100:.1f}%)": p_tail,
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
    macro_feature_indices,
    macro_hidden_dim,
    macro_latent_dim,
    device="cuda"
):
    results = {}

    for name, path in model_paths.items():
        print(f"\n[Point Forecast] Evaluating {name}")

        preds, trues, _ = evaluate_model(
            model_path=path,
            X=X, Y=Y, C=C,
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            hidden=hidden,
            H=H,
            beta=beta,
            macro_feature_indices=macro_feature_indices,
            macro_hidden_dim=macro_hidden_dim,
            macro_latent_dim=macro_latent_dim,
            device=device
        )

        metrics = compute_point_forecast_metrics(preds, trues)
        results[name] = metrics

    return results




def compare_models_probabilistic_nll(
    model_paths,
    X, Y, C,
    latent_dim, cond_dim, hidden, H, beta,
    macro_feature_indices,
    macro_hidden_dim,
    macro_latent_dim,
    device="cuda",
):
    results = {}

    for name, path in model_paths.items():
        print(f"\n[Probabilistic NLL] Evaluating {name}")

        metrics = evaluate_student_t_nll(
            model_path=path,
            X=X, Y=Y, C=C,
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            hidden=hidden,
            H=H,
            beta=beta,
            device=device
        )

        results[name] = metrics

    return results




def run_ablation(
    X, Y, C,
    latent_dim, cond_dim, hidden, H, beta,
    macro_feature_indices,
    macro_hidden_dim,
    macro_latent_dim,
    device="cuda"
):
    print("\n\n======================")
    print("ABALTION: Point Forecast")
    print("======================\n")

    point_results = compare_models_point_forecast(
        model_paths=ablation_model_paths,
        X=X, Y=Y, C=C,
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        hidden=hidden,
        H=H, beta=beta,
        macro_feature_indices=macro_feature_indices,
        macro_hidden_dim=macro_hidden_dim,
        macro_latent_dim=macro_latent_dim,
        device=device
    )

    for name, metrics in point_results.items():
        print(f"\n=== {name} ===")
        print(f"MSE : {metrics['MSE']:.4f}")
        print(f"MAE : {metrics['MAE']:.4f}")
        print(f"RMSE: {metrics['RMSE']:.4f}")


    print("\n\n======================")
    print("ABALTION: Probabilistic NLL")
    print("======================\n")

    prob_results = compare_models_probabilistic_nll(
        model_paths=ablation_model_paths,
        X=X, Y=Y, C=C,
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        hidden=hidden,
        H=H, beta=beta,
        macro_feature_indices=macro_feature_indices,
        macro_hidden_dim=macro_hidden_dim,
        macro_latent_dim=macro_latent_dim,
        device=device
    )

    for name, metrics in prob_results.items():
        print(f"\n=== {name} ===")
        print(f"NLL_mean = {metrics['NLL_mean']:.4f}")
        print(f"NLL_std  = {metrics['NLL_std']:.4f}")

    return point_results, prob_results

####################eval2
import random
import numpy as np
import torch

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def crps_from_samples_fast_1feature(samples_MHD, y_HD, feature_index=0):
    """
    samples_MHD: (M,H,D)
    y_HD:        (H,D)
    returns: (H,) CRPS per horizon for one feature
    O(M log M) per horizon (sort 기반) -> M 크게 가능
    """
    S = np.asarray(samples_MHD)[:, :, feature_index]  # (M,H)
    y = np.asarray(y_HD)[:, feature_index]            # (H,)
    M, H = S.shape

    crps_per_h = np.zeros(H, dtype=float)
    i = np.arange(1, M + 1, dtype=float)

    for h in range(H):
        x = np.sort(S[:, h])
        term1 = np.mean(np.abs(x - y[h]))
        # mean_{i,j}|xi-xj| = (2/(M^2)) * sum_{k=1..M} (2k-M-1)*x_k
        sum_abs = 2.0 * np.sum((2.0 * i - M - 1.0) * x)
        term2 = sum_abs / (M * M)
        crps_per_h[h] = term1 - 0.5 * term2

    return crps_per_h

def load_timevae_for_eval(
    model_path,
    out_dim, cond_dim, hidden, latent_dim, H, beta,
    macro_feature_indices,
    macro_hidden_dim, macro_latent_dim,
    device="cuda",
    macro_encoder_path="macro_encoder.pth",
):
    device = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")

    macro_input_dim = len(macro_feature_indices)
    macro_encoder = MacroEncoder(macro_input_dim, macro_hidden_dim, macro_latent_dim).to(device)
    macro_encoder.load_state_dict(torch.load(macro_encoder_path, map_location=device))
    macro_encoder.eval()
    for p in macro_encoder.parameters():
        p.requires_grad = False

    encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
    decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H).to(device)
    prior   = ConditionalPrior(cond_dim, macro_latent_dim, latent_dim, hidden).to(device)

    model = TimeVAE(encoder, decoder, prior, macro_encoder, latent_dim, beta).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, device
def compute_cvar(return_samples, alpha=0.10):
    """CVaR(alpha): return <= VaR(alpha)의 평균"""
    var_alpha = np.quantile(return_samples, alpha)
    tail = return_samples[return_samples <= var_alpha]
    if len(tail) == 0:
        return var_alpha
    return tail.mean()


def rolling_forward_backtest(
    model_path,
    X, Y, C,
    latent_dim, cond_dim, hidden, H, beta,
    macro_feature_indices,
    macro_hidden_dim=128,
    macro_latent_dim=32,
    device="cuda",
    macro_encoder_path="macro_encoder.pth",
    # CRPS 설정
    crps_feature_index=0,
    num_samples=500,
    z_shrink=1.0,
    sample_from_student_t=True,
    seed=0,
):
    """
    Rolling-forward evaluation
    X: (N,L,D), Y: (N,H,D)
    """
    set_seed(seed)

    X = np.asarray(X); Y = np.asarray(Y); C = np.asarray(C)
    N, L, D = X.shape

    # ------------------------------
    # Load model
    # ------------------------------
    model, dev = load_timevae_for_eval(
        model_path=model_path,
        out_dim=D, cond_dim=cond_dim, hidden=hidden, latent_dim=latent_dim, H=H, beta=beta,
        macro_feature_indices=macro_feature_indices,
        macro_hidden_dim=macro_hidden_dim, macro_latent_dim=macro_latent_dim,
        device=device,
        macro_encoder_path=macro_encoder_path,
    )

    # ------------------------------
    # Accumulators
    # ------------------------------
    sse = 0.0
    sae = 0.0
    n_elem = 0

    nll_sum = 0.0
    nll_count = 0

    crps_sum_per_h = np.zeros(H, dtype=float)
    crps_count = 0

    coverage_sum = 0.0
    sharpness_sum = 0.0
    coverage_count = 0

    # --- CVaR holder ---
    cvar_10 = None

    # ======================================================
    # Rolling loop
    # ======================================================
    with torch.no_grad():
        for i in range(N):
            x = torch.tensor(X[i:i+1]).float().to(dev)     # (1,L,D)
            y = torch.tensor(Y[i:i+1]).float().to(dev)     # (1,H,D)
            c = torch.tensor(C[i:i+1]).float().to(dev)     # (1,cond)

            # ---- posterior ----
            mu_q, logvar_q = model.encoder(x, c)
            mean, dist = model.decoder(mu_q, c)            # (1,H,D)

            # ===== Point Forecast Metrics =====
            diff = (mean - y).cpu().numpy()
            sse += float(np.sum(diff**2))
            sae += float(np.sum(np.abs(diff)))
            n_elem += diff.size

            # ===== NLL =====
            nll_sum += -dist.log_prob(y).mean().item()
            nll_count += 1

            # ===== Scenario sampling =====
            M = int(num_samples)
            std_q = torch.exp(0.5 * logvar_q)

            eps = torch.randn((M, latent_dim), device=dev)
            z = mu_q.expand(M, -1) + z_shrink * std_q.expand(M, -1) * eps
            c_rep = c.expand(M, -1)

            mean_s, dist_s = model.decoder(z, c_rep)       # (M,H,D)
            y_s = dist_s.rsample() if sample_from_student_t else mean_s
            samples = y_s.cpu().numpy()                    # (M,H,D)

            # ===== CRPS =====
            crps_per_h = crps_from_samples_fast_1feature(
                samples_MHD=samples,
                y_HD=Y[i],
                feature_index=crps_feature_index
            )
            crps_sum_per_h += crps_per_h
            crps_count += 1

            # ===== Coverage / Sharpness =====
            cinfo = compute_coverage_and_sharpness(
                scenario_samples=samples,
                true_future=Y[i],
                feature_index=crps_feature_index,
                lower_q=10,
                upper_q=90
            )
            coverage_sum += cinfo["Coverage_80%"]
            sharpness_sum += cinfo["Sharpness_80%"]
            coverage_count += 1

            # ======================================================
            # ====== CVaR(10%) — 마지막 시점에서만 계산 ======
            # ======================================================
            if i == N - 1:
                # 마지막 horizon 의 scenario 값 (raw scaled)
                y_last = samples[:, -1, crps_feature_index]     # (M,)

                # 현재 시점(X[i])의 마지막 관측값
                current = X[i][-1, crps_feature_index]          # scaler 이후 값

                # scaled return
                returns = (y_last / current) - 1.0              # (M,)

                cvar_10 = float(compute_cvar(returns, alpha=0.10))

    # ------------------------------
    # Final Metrics
    # ------------------------------
    mse = sse / max(1, n_elem)
    rmse = float(np.sqrt(mse))
    mae = sae / max(1, n_elem)

    nll_mean = nll_sum / max(1, nll_count)

    crps_per_h = crps_sum_per_h / max(1, crps_count)
    crps_mean = float(np.mean(crps_per_h))

    coverage_final = coverage_sum / max(1, coverage_count)
    sharpness_final = sharpness_sum / max(1, coverage_count)

    # ====== 반환 ======
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse,
        "NLL_mean": nll_mean,
        "CRPS_mean": crps_mean,
        "CRPS_per_h": crps_per_h,
        "Coverage_80%": coverage_final,
        "Sharpness_80%": sharpness_final,
        "CVaR_10%": cvar_10,          # ★ 핵심 추가
        "settings": {
            "num_samples": int(num_samples),
            "z_shrink": float(z_shrink),
            "student_t_sample": bool(sample_from_student_t),
            "seed": int(seed),
            "crps_feature_index": int(crps_feature_index),
        }
    }


