# ===========================================
# scenario_eval_utils.py
# (Scenario generation + Evaluation + Plotting)
# ===========================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import gaussian_kde
import math
import pandas as pd

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
    macro_feature_indices,
    macro_hidden_dim, macro_latent_dim,
    device="cuda"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dim = X.shape[-1]

    # macro encoder (train과 동일 세팅)
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

    nll_list = []
    with torch.no_grad():
        for i in range(len(X)):
            x = torch.tensor(X[i:i+1]).float().to(device)
            y = torch.tensor(Y[i:i+1]).float().to(device)
            c = torch.tensor(C[i:i+1]).float().to(device)

            macro_x = x[:, :, macro_feature_indices].permute(0, 2, 1)

            # train과 같은 경로로 recon_loss = -log_prob.mean()
            loss, recon, kl, mean, z, prior_stats = model(
                x, c, macro_x,
                y=y,
                use_prior_sampling_if_no_y=False
            )
            nll_list.append(recon.item())

    nll_arr = np.array(nll_list, dtype=np.float32)
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

def plot_baseline_fanchart_recent(
    true_seq_full,          # df_scaled.values 전체 (scaled history)
    pred_seq_last,          # preds[-1] (마지막 chunk의 H-step 예측 median)
    scenario_samples,       # scenario_samples (M, H, D)
    df_index,               # df_scaled.index (DatetimeIndex)
    feature_index=0,
    history=60              # 최근 5년(=60 months)
):
    """
    Baseline Fan Chart (최근 5년 + 예측 12개월)
    - 과거 데이터와 예측 데이터가 '연속된 시계열'로 자연스럽게 이어짐
    """

    # -----------------------------
    # 1) 최근 히스토리 추출
    # -----------------------------
    past_values = true_seq_full[-history:, feature_index]
    past_dates = df_index[-history:]

    # -----------------------------
    # 2) 예측 구간 날짜 만들기
    # -----------------------------
    last_date = df_index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=len(pred_seq_last),
        freq='MS'
    )

    # -----------------------------
    # 3) Fan Chart 구간 계산
    # -----------------------------
    samples = scenario_samples[:, :, feature_index]     # (num_samples, H)

    p10 = np.percentile(samples, 10, axis=0)
    p90 = np.percentile(samples, 90, axis=0)
    p25 = np.percentile(samples, 25, axis=0)
    p75 = np.percentile(samples, 75, axis=0)
    median_pred = np.median(samples, axis=0)

    median_pred = median_pred - median_pred[0] + past_values[-1]
    p10 = p10 - p10[0] + past_values[-1]
    p25 = p25 - p25[0] + past_values[-1]
    p75 = p75 - p75[0] + past_values[-1]
    p90 = p90 - p90[0] + past_values[-1]
    # -----------------------------
    # 4) 그래프 그리기
    # -----------------------------
    plt.figure(figsize=(14, 6))

    # 과거 구간
    plt.plot(past_dates, past_values, label="History (True)", color="blue")

    # 예측 구간: P10–P90
    plt.fill_between(future_dates, p10, p90, color="skyblue", alpha=0.3, label="P10–P90")

    # 예측 구간: P25–P75
    plt.fill_between(future_dates, p25, p75, color="orange", alpha=0.5, label="P25–P75")

    # 예측 median
    plt.plot(future_dates, median_pred, "--", color="darkorange", linewidth=2, label="Scenario median")

    # 마지막 히스토리 포인트 강조
    plt.scatter([past_dates[-1]], [past_values[-1]], color="red", s=60, zorder=5)

    # -----------------------------
    # 5) X축 날짜 포맷
    # -----------------------------
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.title("Baseline Fan Chart (Recent 5 Years + Next 12 Months)", fontsize=14)
    plt.ylabel("Scaled value")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_forecast_zoom(
    pred_seq_last,         # preds[-1] / Rolling_forecast 라도 가능
    scenario_samples,      # scenario_samples (M, H, D)
    df_index,              # df_scaled.index
    feature_index=0,
):
    """
    예측 12개월만 확대해서 (fan chart + median) 시각화
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import matplotlib.dates as mdates

    H = pred_seq_last.shape[0]

    # 날짜 생성
    last_date = df_index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=H,
        freq='MS'
    )

    # Fan chart data
    samples = scenario_samples[:, :, feature_index]
    p10 = np.percentile(samples, 10, axis=0)
    p90 = np.percentile(samples, 90, axis=0)
    p25 = np.percentile(samples, 25, axis=0)
    p75 = np.percentile(samples, 75, axis=0)
    median_pred = np.median(samples, axis=0)

    # Plot
    plt.figure(figsize=(8, 5))

    # Fan chart
    plt.fill_between(future_dates, p10, p90, color="skyblue", alpha=0.3, label="P10–P90")
    plt.fill_between(future_dates, p25, p75, color="orange", alpha=0.5, label="P25–P75")
    plt.plot(future_dates, median_pred, "--", color="darkorange", linewidth=2, label="Scenario Median")

    # Format
    plt.title("Zoom-in: Next 12 Months Forecast", fontsize=13)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_kde_with_stats(scenario_samples_dict, feature_index=0, horizon_idx=-1):
    """
    KDE Density Curve + Scenario Mean/Std Table (자동 생성)
    """

    plt.figure(figsize=(12, 6))

    stats = []   # mean / std 저장

    for name, samples in scenario_samples_dict.items():
        vals = samples[:, horizon_idx, feature_index]

        # KDE
        kde = gaussian_kde(vals)
        x_grid = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 300)
        plt.plot(x_grid, kde(x_grid), linewidth=2, label=name)

        # 통계 저장
        stats.append({
            "Scenario": name,
            "Mean (μ)": np.mean(vals),
            "Std (σ)": np.std(vals),
            "5%": np.percentile(vals, 5),
            "95%": np.percentile(vals, 95)
        })

    # 그래프 스타일
    plt.title("Scenario Comparison — Density Curve (KDE)")
    plt.xlabel("Predicted Export Level (Scaled or Raw)")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ====== 테이블 출력 ======
    df_stats = pd.DataFrame(stats)
    print("\n=== Scenario Mean / Std / Quantile Table ===")
    print(df_stats.to_string(index=False))

    return df_stats



def plot_df_scale_shift(df_dict, scale_dict):
    """
    시나리오별 df / scale 변화 비교
    df_dict: {"Baseline": df_arr, "ExRate": df_arr, ...}
    scale_dict: 동일 구조
    """

    horizons = np.arange(1, len(list(df_dict.values())[0]) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # ---- df plot ----
    for name, df_arr in df_dict.items():
        axs[0].plot(horizons, df_arr, label=name)
    axs[0].set_title("df (Degrees of Freedom) Comparison")
    axs[0].set_xlabel("Horizon")
    axs[0].set_ylabel("df")
    axs[0].grid(alpha=0.3)
    axs[0].legend()

    # ---- scale plot ----
    for name, sc_arr in scale_dict.items():
        axs[1].plot(horizons, sc_arr, label=name)
    axs[1].set_title("Scale (Uncertainty Level) Comparison")
    axs[1].set_xlabel("Horizon")
    axs[1].set_ylabel("Scale")
    axs[1].grid(alpha=0.3)
    axs[1].legend()

    plt.tight_layout()
    plt.show()







