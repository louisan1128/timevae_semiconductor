# ===========================================
# main.py — 전체 파이프라인 실행
# ===========================================

import numpy as np

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
    student_t_nll_torch,
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
# Condition columns (must exist in merged df)
# -------------------------------
condition_raw_cols = [
    "Exchange Rate",
    "CAPEX",
    "PMI",
    "CLI",
    "ISM",
]

# macro.csv base columns (preprocess에서 feature engineering해서 6개로 만듦)
MACRO_COLS = [
    "PMI",
    "GS10",
    "M2SL",
    "UNRATE",
    "CPIAUCSL",
    "INDPRO",
]

# ===========================================
# Main
# ===========================================
if __name__ == "__main__":

    print("========== 1) Preprocessing ==========")

    X, Y, C, MACRO_X, scaler, df_raw, df_scaled, macro_feature_indices, macro_feat_cols = preprocess(
        csv_path="data.csv",
        macro_csv_path="macro.csv",
        condition_raw_cols=condition_raw_cols,
        macro_cols=MACRO_COLS,
        L=L,
        H=H,
        # macro_scaler_path="macro_scaler.pkl",  # 있으면 켜기(없으면 주석)
    )

    print("X:", X.shape, "Y:", Y.shape, "C:", C.shape, "MACRO_X:", MACRO_X.shape)
    print("======================================\n")

    # =======================================
    # 2) Train model
    # =======================================
    print("========== 2) Training ==========")

    model = train_model(
        X, Y, C, MACRO_X,
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        hidden=HIDDEN,
        H_len=H,
        beta=BETA,
        lr=LR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        macro_hidden_dim=MACRO_HIDDEN_DIM,
        macro_latent_dim=MACRO_LATENT_DIM,
        macro_encoder_path="macro_encoder.pth",
    )

    print("=================================\n")

    # =======================================
    # 5) Posterior Evaluation
    # =======================================
    print("========== 5) Posterior Evaluation ==========")

    preds, trues, mse_eval = evaluate_model(
        model_path="timevae_ctvae_prior.pth",
        X=X, Y=Y, C=C, MACRO_X=MACRO_X,
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        hidden=HIDDEN,
        H=H,
        beta=BETA,
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

    last_truth_raw = df_raw.iloc[-1]
    print("===== Last RAW sample =====")
    print(last_truth_raw)

    scenario_cond_raw = {
        "Exchange Rate": 1388.91,
        "CAPEX": 93163.0,
        "PMI": 48.0,
        "CLI": 100.27,
        "ISM": 51.4,
    }

    cols = list(df_raw.columns)
    full_raw = [scenario_cond_raw.get(col, last_truth_raw[col]) for col in cols]
    scaled_full = scaler.transform([full_raw])[0]

    scenario_cond_scaled = np.array(
        [scaled_full[cols.index(col)] for col in condition_raw_cols],
        dtype=np.float32
    )

    scenario_samples = scenario_predict_local(
        model_path="timevae_ctvae_prior.pth",
        X_last=X[-1],                     # (L,D)
        macro_x_last=MACRO_X[-1],          # (6,L)
        cond_true=C[-1],
        cond_scenario=scenario_cond_scaled,
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        hidden=HIDDEN,
        H=H,
        beta=BETA,
        num_samples=50,
        z_shrink=0.5,
        macro_hidden_dim=MACRO_HIDDEN_DIM,
        macro_latent_dim=MACRO_LATENT_DIM,
        device=DEVICE,
        sample_from_student_t=True
    )

    # =======================================
    # 7) Fan Chart
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

    # Rolling forecast
    forecast_full = rolling_posterior_forecast(
        "timevae_ctvae_prior.pth",
        X, C, MACRO_X,
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        hidden=HIDDEN,
        H=H,
        beta=BETA,
        macro_hidden_dim=MACRO_HIDDEN_DIM,
        macro_latent_dim=MACRO_LATENT_DIM,
        device=DEVICE
    )

    # 마지막 시점 posterior scenario (mean-based)
    samples = posterior_scenario(
        model_path="timevae_ctvae_prior.pth",
        X_last=X[-1],
        C_last=C[-1],
        macro_x_last=MACRO_X[-1],
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        hidden=HIDDEN,
        H=H,
        beta=BETA,
        macro_hidden_dim=MACRO_HIDDEN_DIM,
        macro_latent_dim=MACRO_LATENT_DIM,
        num_samples=30,
        shrink=0.1,
        device=DEVICE
    )

    true_full = Y[:, 0, :]

    plot_full_forecast_and_scenario(
        true_full=true_full,
        forecast_full=forecast_full,
        scenario_samples=scenario_samples,
        feature_index=0,
        H=H
    )

    # Metrics
    point_metrics = compute_point_forecast_metrics(preds, trues)
    print("=========== Completed 2nd! ===========")

    print("=== Point Forecast Metrics ===")
    for k, v in point_metrics.items():
        print(f"{k} : {v}")

    nll_metrics = evaluate_student_t_nll(
        model_path="timevae_ctvae_prior.pth",
        X=X, Y=Y, C=C, MACRO_X=MACRO_X,
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        hidden=HIDDEN,
        H=H,
        beta=BETA,
        device=DEVICE,
        macro_hidden_dim=MACRO_HIDDEN_DIM,
        macro_latent_dim=MACRO_LATENT_DIM,
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

    current_level_scaled = X[-1][-1, :]
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
