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
    compare_models_point_forecast,
    compare_models_probabilistic_nll

)

# -------------------------------
# Hyperparameters
# -------------------------------
L = 36
H = 12

LATENT_DIM = 32
COND_DIM = 5
HIDDEN = 128

POLY_ORDER = 2
N_FOURIER = 3

BETA = 1.0
LR = 1e-3
EPOCHS = 80          # rolling forward 재학습 시 epoch 많으면 오래 걸림
BATCH_SIZE = 32

DEVICE = "cuda"


# -------------------------------
# Condition columns (raw)
# -------------------------------
condition_raw_cols = [
    "Exchange Rate",
    "CAPEX",
    "Global Manufacturing PMI",
    "OECD CLI",
    "U.S. ISM Manufacturing New Orders Index",
]


# ===========================================
# Main
# ===========================================
if __name__ == "__main__":

    print("========== 1) Preprocessing ==========")

    X, Y, C, scaler, df_raw, df_scaled = preprocess(
        "data.csv",
        condition_raw_cols,
        L,
        H
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
        device=DEVICE
    )

    print("=================================\n")


    # =======================================
    # 3) Rolling Backtest (고정 모델)
    # =======================================
    print("========== 3) Rolling Backtest ==========")

    mse_back = rolling_backtest(
        model_path="timevae_ctvae_prior.pth",
        X=X, Y=Y, C=C,
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        hidden=HIDDEN,
        H=H,
        beta=BETA,
        device=DEVICE
    )

    print(f"Rolling Backtest MSE: {mse_back:.6f}")
    print("=================================\n")


    # =======================================
    # 4) Rolling Forward Test (매번 재학습)
    # =======================================
    print("========== 4) Rolling Forward Test ==========")

    mse_forward = rolling_forward_test(
        X, Y, C,
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        hidden=HIDDEN,
        H=H,
        beta=BETA,
        lr=LR,
        epochs=10,              # Rolling forward는 epoch을 줄이지 않으면 너무 오래 걸림
        batch_size=BATCH_SIZE,
        device=DEVICE
    )

    print(f"Rolling Forward Test MSE: {mse_forward:.6f}")
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
        device=DEVICE
    )

    print(f"Posterior Recon MSE: {mse_eval:.6f}")
    print("=================================\n")

    print(f"Rolling Backtest MSE: {mse_back:.6f}")
    print(f"Rolling Forward Test MSE: {mse_forward:.6f}")
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
        "Exchange Rate": 1393.41,
        "CAPEX": 96404.0,
        "Global Manufacturing PMI": 52.4,
        "OECD CLI": 100.35,
        "U.S. ISM Manufacturing New Orders Index": 48.9,
    }

    # raw → scaled 변환
    full_raw = [
        scenario_cond_raw[col] if col in scenario_cond_raw else last_truth_raw[col]
        for col in df_raw.columns
    ]

    scaled_full = scaler.transform([full_raw])[0]

    scenario_cond_scaled = np.array([
        scaled_full[df_raw.columns.get_loc(col)]
        for col in condition_raw_cols
    ])

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
        z_shrink=0.1,
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
        feature_index=0    # Export Index
    )

    plot_fanchart_long(
        true_seq_full=df_scaled.values,    # 전체 스케일된 시계열
        pred_seq_last=preds[-1],           # 마지막 chunk prediction
        scenario_samples=scenario_samples,
        feature_index=0,
        history=60                         # 5년치
    )


    print("=========== Completed! ===========")

    # 1) Rolling forecast (파란선)
    forecast_full = rolling_posterior_forecast(
        "timevae_ctvae_prior.pth",
        X, C,
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        hidden=HIDDEN,
        H=H,
        beta=BETA,
        device=DEVICE
    )

    # 2) 마지막 시점 scenario (빨간선)
    scenario_samples = posterior_scenario(
        "timevae_ctvae_prior.pth",
        X_last=X[-1],
        C_last=C[-1],
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        hidden=HIDDEN,
        H=H,
        beta=BETA,
        num_samples=50,
        shrink=0.2,
        device=DEVICE
    )

    # 3) true 전체 시계열 (각 chunk의 첫 y가 true future 1-step)
    true_full = Y[:, 0, :]

    # 4) Plot
    plot_full_forecast_and_scenario(
        true_full=true_full,
        forecast_full=forecast_full,
        scenario_samples=scenario_samples,
        feature_index=0,
        H=H
    )


    point_metrics  = compute_point_forecast_metrics(preds, trues)
    print("=== Point Forecast Metrics ===")
    for k, v in point_metrics.items():
        print(f"{k} : {v}")

    
    nll_metrics = evaluate_student_t_nll(
        model_path="timevae_ctvae_prior.pth",
        X=X, Y=Y, C=C,
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        hidden=HIDDEN,
        H=H,
        beta=BETA,
        device=DEVICE
    )

    print("=== Probabilistic (Student-t) Metrics ===")
    print("NLL_mean:", nll_metrics["NLL_mean"])
    print("NLL_std :", nll_metrics["NLL_std"])


    coverage = compute_coverage_and_sharpness(
        scenario_samples,
        true_future=Y[-1],
        feature_index=0,   # export index
        lower_q=10,
        upper_q=90
    )

    print("=== Coverage / Sharpness ===")
    print(coverage)

    crps = compute_crps_from_samples(
        scenario_samples,
        true_future=Y[-1],
        feature_index=0
    )

    print("=== CRPS ===")
    print(crps)

    current_level = X[-1][-1, 0]   # 마지막 시점 export (예: feature 0)

    risk = compute_risk_metrics(
        scenario_samples,
        current_level=current_level,
        feature_index=0,
        horizon_idx=-1,
        tail_threshold=-0.1,   # -10%
        alpha=0.10
    )

    print("=== Risk Metrics ===")
    for k, v in risk.items():
        print(f"{k}: {v}")



