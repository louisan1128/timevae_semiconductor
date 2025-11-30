# ===========================================
# main.py — 전체 파이프라인 실행 (FIXED)
# ===========================================

import numpy as np
import pandas as pd

from data_train import (
    preprocess,
    train_model,
    rolling_backtest,
    rolling_forward_test,
)

from scenario_eval import (
    evaluate_model,
    scenario_predict_local,
    plot_fanchart,
    plot_fanchart_long,
    rolling_posterior_forecast,
    plot_full_forecast_and_scenario,
    posterior_scenario,

    compute_point_forecast_metrics,
    compute_coverage_and_sharpness,
    compute_crps_from_samples,
    evaluate_student_t_nll,
    compute_risk_metrics,
)

# -------------------------------
# Hyperparameters
# -------------------------------
L = 36
H = 12

LATENT_DIM = 32
COND_DIM = 5
HIDDEN = 128

BETA = 1.0
LR = 1e-3
EPOCHS = 150
BATCH_SIZE = 32

DEVICE = "cuda"

MACRO_HIDDEN_DIM = 128
MACRO_LATENT_DIM = 32

# -------------------------------
# Condition columns (raw)
# -------------------------------
condition_raw_cols = [
    "Exchange Rate",
    "CAPEX",
    "PMI",
    "CLI",
    "ISM",
]

MACRO_COLS = [
    "PMI",
    "GS10",
    "M2SL",
    "UNRATE",
    "CPIAUCSL",
    "INDPRO",
]


def _clean_numeric_series(s: pd.Series) -> pd.Series:
    # "1,388.90" 같은 문자열/공백 안전 처리
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def make_scaled_condition_vector(
    *,
    last_truth_raw: pd.Series,
    scenario_cond_raw: dict,
    df_scaled: pd.DataFrame,
    scaler,
    condition_raw_cols: list,
) -> np.ndarray:
    """
    preprocess와 동일한 규칙(숫자 coercion + CAPEX log1p)로
    raw scenario row -> scaled row -> condition vector를 만든다.
    - feature-name warning 제거 (DataFrame으로 transform)
    - object dtype 제거
    - CAPEX 변환 불일치 버그 제거
    """
    cols = list(df_scaled.columns)

    # last row를 dict로 만들고 scenario 값으로 overwrite
    row = last_truth_raw.to_dict()
    row.update(scenario_cond_raw)

    # scaler가 fit된 컬럼 순서로 DataFrame 구성
    row_df = pd.DataFrame([row], columns=cols)

    # numeric coercion
    row_df = row_df.apply(_clean_numeric_series, axis=0).fillna(0.0)

    # preprocess와 동일: CAPEX log1p(음수 방어)
    if "CAPEX" in row_df.columns:
        cap = row_df["CAPEX"].to_numpy(dtype=np.float64)
        cap = np.clip(cap, 0.0, None)
        row_df["CAPEX"] = np.log1p(cap)

    scaled_full = scaler.transform(row_df)[0]

    cond_vec = np.array(
        [scaled_full[cols.index(col)] for col in condition_raw_cols],
        dtype=np.float32,
    )
    return cond_vec


# ===========================================
# Main
# ===========================================
if __name__ == "__main__":

    print("========== 1) Preprocessing ==========")

    X, Y, C, scaler, df_raw, df_scaled, macro_feature_indices = preprocess(
        csv_path="data.csv",
        macro_csv_path="macro.csv",
        condition_raw_cols=condition_raw_cols,
        macro_cols=MACRO_COLS,
        L=L,
        H=H,
    )

    print("X:", X.shape, "Y:", Y.shape, "C:", C.shape)
    print("======================================\n")

    # =======================================
    # 2) Train model
    # =======================================
    print("========== 2) Training ==========")

    model = train_model(
        X, Y, C,
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        hidden=HIDDEN,
        H_len=H,
        beta=BETA,
        lr=LR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        macro_feature_indices=macro_feature_indices,
        macro_latent_dim=MACRO_LATENT_DIM,
        macro_hidden_dim=MACRO_HIDDEN_DIM,
    )

    print("=================================\n")

    # =======================================
    # 5) Posterior Evaluation
    # =======================================
    print("========== 5) Posterior Evaluation ==========")

    preds, trues, mse_eval = evaluate_model(
        model_path="timevae_ctvae_prior.pth",
        X=X, Y=Y, C=C,
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        hidden=HIDDEN,
        H=H,
        beta=BETA,
        macro_feature_indices=macro_feature_indices,
        macro_hidden_dim=MACRO_HIDDEN_DIM,
        macro_latent_dim=MACRO_LATENT_DIM,
        device=DEVICE
    )

    print(f"Posterior Recon MSE: {mse_eval:.6f}")
    print("=================================\n")

    # =======================================
    # 6) Scenario Generation
    # =======================================
    print("========== 6) Scenario Forecasting ==========")

    # 실제 마지막 month의 RAW 값
    last_truth_raw = df_raw.iloc[-1]
    print("===== Last RAW sample =====")
    print(last_truth_raw)

    # 시나리오 조건 (RAW 형태)
    scenario_cond_raw = {
        "Exchange Rate": 1388.91,
        "CAPEX": 93163.0,
        "PMI": 48.0,
        "CLI": 100.27,
        "ISM": 51.4,
    }

    # ✅ FIXED: safe raw->scaled condition vector (feature-name warning / object dtype / CAPEX log1p 일치)
    scenario_cond_scaled = make_scaled_condition_vector(
        last_truth_raw=last_truth_raw,
        scenario_cond_raw=scenario_cond_raw,
        df_scaled=df_scaled,
        scaler=scaler,
        condition_raw_cols=condition_raw_cols,
    )

    # 시나리오 샘플링
    scenario_samples = scenario_predict_local(
        model_path="timevae_ctvae_prior.pth",
        X_last=X[-1],
        cond_true=C[-1],
        cond_scenario=scenario_cond_scaled,
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        hidden=HIDDEN,
        H=H,
        beta=BETA,
        num_samples=50,
        z_shrink=0.5,
        macro_feature_indices=macro_feature_indices,
        macro_hidden_dim=MACRO_HIDDEN_DIM,
        macro_latent_dim=MACRO_LATENT_DIM,
        device=DEVICE
    )

    # =======================================
    # 7) Fan Chart 출력
    # =======================================
    print("========== 7) Plotting Fan Chart ==========")

    plot_fanchart(
        true_seq=trues[-1],
        pred_seq=preds[-1],
        scenario_samples=scenario_samples,
        feature_index=0
    )

    plot_fanchart_long(
        true_seq_full=df_scaled.values,
        pred_seq_last=preds[-1],
        scenario_samples=scenario_samples,
        feature_index=0,
        history=60
    )

    print("=========== Completed 1st! ===========")

    # 1) Rolling forecast (파란선)
    forecast_full = rolling_posterior_forecast(
        "timevae_ctvae_prior.pth",
        X,
        C,
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        hidden=HIDDEN,
        H=H,
        beta=BETA,
        macro_feature_indices=macro_feature_indices,
        macro_hidden_dim=MACRO_HIDDEN_DIM,
        macro_latent_dim=MACRO_LATENT_DIM,
        device=DEVICE
    )

    # 2) 마지막 시점 posterior-scenario (빨간선)
    samples_posterior = posterior_scenario(
        model_path="timevae_ctvae_prior.pth",
        X_last=X[-1],
        C_last=C[-1],
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        hidden=HIDDEN,
        H=H,
        beta=BETA,
        macro_feature_indices=macro_feature_indices,
        macro_hidden_dim=MACRO_HIDDEN_DIM,
        macro_latent_dim=MACRO_LATENT_DIM,
        num_samples=30,
        shrink=0.1,
        device=DEVICE
    )

    # 3) true 전체 시계열 (각 chunk의 첫 y가 true future 1-step)
    true_full = Y[:, 0, :]

    # 4) Plot (scenario는 “scenario_predict_local” 결과로)
    plot_full_forecast_and_scenario(
        true_full=true_full,                 # (N, D)
        forecast_full=forecast_full,         # (N, D)
        scenario_samples=scenario_samples,   # (M, H, D)
        feature_index=0,
        H=H
    )

    print("=========== Completed 2nd! ===========")

    # =======================================
    # Metrics
    # =======================================
    point_metrics = compute_point_forecast_metrics(preds, trues)
    print("=== Point Forecast Metrics ===")
    for k, v in point_metrics.items():
        print(f"{k} : {v}")

    # ✅ FIXED: NLL eval should match macro_feature_indices too
    nll_metrics = evaluate_student_t_nll(
        model_path="timevae_ctvae_prior.pth",
        X=X,
        Y=Y,
        C=C,
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        hidden=HIDDEN,
        H=H,
        beta=BETA,
        macro_feature_indices=macro_feature_indices,
        macro_hidden_dim=MACRO_HIDDEN_DIM,
        macro_latent_dim=MACRO_LATENT_DIM,
        device=DEVICE
    )

    print("=== Probabilistic (Student-t) Metrics ===")
    print("NLL_mean:", nll_metrics["NLL_mean"])
    print("NLL_std :", nll_metrics["NLL_std"])

    coverage = compute_coverage_and_sharpness(
        scenario_samples,
        true_future=Y[-1],
        feature_index=0,
        lower_q=10,
        upper_q=90
    )

    print("=== Coverage / Sharpness ===")
    for k, v in coverage.items():
        print(f"{k}: {float(v):.4f}")

    crps = compute_crps_from_samples(
        scenario_samples,
        true_future=Y[-1],
        feature_index=0
    )

    print("=== CRPS ===")
    print(f"CRPS_mean: {crps['CRPS_mean']:.4f}")
    print("CRPS_per_h:", np.round(crps["CRPS_per_h"], 4))

    current_level_scaled = X[-1][-1, :]  # scaled current level

    risk = compute_risk_metrics(
        scenario_samples=scenario_samples,
        current_level_scaled=current_level_scaled,
        scaler=scaler,
        feature_index=0,
        horizon_idx=-1,
        tail_threshold_raw=-0.10,
        alpha=0.10
    )

    print("=== Risk Metrics ===")
    for k, v in risk.items():
        print(f"{k}: {v}")
        
def debug_last_step_distribution(model, X, C, macro_feature_indices, Y=None, tag=""):
    device = next(model.parameters()).device
    x = torch.tensor(X[-1:]).float().to(device)          # (1,L,D)
    c = torch.tensor(C[-1:]).float().to(device)          # (1,cond_dim)
    macro_x = x[:, :, macro_feature_indices].permute(0, 2, 1)  # (1,macro_dim,L)

    with torch.no_grad():
        mu_q, logvar_q = model.encoder(x, c)
        zq = mu_q
        mean_q, dist_q = model.decoder(zq, c)

        # prior 쪽도 비교(shift가 먹는지)
        z_macro_out = model.macro_encoder(macro_x)
        z_macro = z_macro_out[0] if isinstance(z_macro_out, (tuple, list)) else z_macro_out
        mu_p, logvar_p = model.prior(c, z_macro)
        zp = mu_p
        mean_p, dist_p = model.decoder(zp, c)

        def _stat(name, dist):
            df = dist.df.detach().cpu().numpy().ravel()
            sc = dist.scale.detach().cpu().numpy().ravel()
            print(f"[{tag}] {name}: df median={np.median(df):.2f} (p10={np.percentile(df,10):.2f}, p90={np.percentile(df,90):.2f}) | "
                  f"scale median={np.median(sc):.4f} (p10={np.percentile(sc,10):.4f}, p90={np.percentile(sc,90):.4f})")

        print(f"[{tag}] ||mu_q-mu_p||_2 =", float(torch.norm(mu_q - mu_p, p=2)))
        print(f"[{tag}] mean(std_q) =", float(torch.exp(0.5*logvar_q).mean()))
        _stat("posterior(z=mu_q)", dist_q)
        _stat("prior(z=mu_p)", dist_p)

        if Y is not None:
            y = torch.tensor(Y[-1:]).float().to(device)
            nll_q = -dist_q.log_prob(y).mean().item()
            nll_p = -dist_p.log_prob(y).mean().item()
            print(f"[{tag}] NLL using z=mu_q: {nll_q:.4f} | using z=mu_p: {nll_p:.4f}")

debug_last_step_distribution(model, X, C, macro_feature_indices, Y=Y, tag="LAST")
