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
    compare_models_probabilistic_nll,
    run_ablation

)

# -------------------------------
# Hyperparameters
# -------------------------------
L = 36
H = 12

LATENT_DIM = 32

COND_DIM = 4
HIDDEN = 128

POLY_ORDER = 2
N_FOURIER = 3

BETA = 1.0
LR = 1e-3
EPOCHS = 150         # rolling forward 재학습 시 epoch 많으면 오래 걸림
BATCH_SIZE = 32

DEVICE = "cuda"

MACRO_HIDDEN_DIM = 128  # macro encoder hidden dim
MACRO_LATENT_DIM = 32   # macro encoder latent dim
# -------------------------------
# Condition columns (raw)
# -------------------------------
condition_raw_cols = [
    "Exchange Rate",
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
    )

    print("=================================\n")


    # =======================================
    # 3) Rolling Backtest (고정 모델)
    # =======================================
    # print("========== 3) Rolling Backtest ==========")

    # mse_back = rolling_backtest(
    #     model_path="timevae_ctvae_prior.pth",
    #     X=X, Y=Y, C=C,
    #     latent_dim=LATENT_DIM,
    #     cond_dim=COND_DIM,
    #     hidden=HIDDEN,
    #     H=H,
    #     beta=BETA,
    #     macro_feature_indices=macro_feature_indices,
    #     device=DEVICE,
    # )

    # print(f"Rolling Backtest MSE: {mse_back:.6f}")
    # print("=================================\n")


    # =======================================
    # 4) Rolling Forward Test (매번 재학습)
    # =======================================
    # print("========== 4) Rolling Forward Test ==========")

    # mse_forward = rolling_forward_test(
    #     X, Y, C,
    #     latent_dim=LATENT_DIM,
    #     cond_dim=COND_DIM,
    #     hidden=HIDDEN,
    #     H=H,
    #     beta=BETA,
    #     lr=LR,
    #     epochs=10,
    #     batch_size=BATCH_SIZE,
    #     macro_feature_indices=macro_feature_indices,
    #     device=DEVICE,
    # )

    # print(f"Rolling Forward Test MSE: {mse_forward:.6f}")
    # print("=================================\n")


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

    # print(f"Rolling Backtest MSE: {mse_back:.6f}")
    # print(f"Rolling Forward Test MSE: {mse_forward:.6f}")
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
        "PMI": 48.0,
        "CLI": 100.27,
        "ISM": 51.4,
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
        z_shrink=0.5,
        macro_feature_indices=macro_feature_indices,   # ★ 추가
        macro_hidden_dim=MACRO_HIDDEN_DIM,             # ★ 추가
        macro_latent_dim=MACRO_LATENT_DIM,             # ★ 추가
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

    # 2) 마지막 시점 scenario (빨간선)
    samples = posterior_scenario(
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

    # 4) Plot
    plot_full_forecast_and_scenario(
        true_full=true_full,            # shape (N, D)
        forecast_full=forecast_full,    # rolling forecast (N, D)
        scenario_samples=scenario_samples,  # shape (num_samples, H, D)
        feature_index=0,
        H=H
    )

    point_metrics  = compute_point_forecast_metrics(preds, trues)
    print("=========== Completed 2nd! ===========")

    


    print("=== Point Forecast Metrics ===")
    for k, v in point_metrics.items():
        print(f"{k} : {v}")

    
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






    current_level_scaled = X[-1][-1, :]       # Scaled current level

    risk = compute_risk_metrics(
        scenario_samples=scenario_samples,
        current_level_scaled=current_level_scaled,
        scaler=scaler,
        feature_index=0,           # export
        horizon_idx=-1,            # 마지막 horizon
        tail_threshold_raw=-0.10,  # -10% drop
        alpha=0.10                 # 10% VaR
    )

    print("=== Risk Metrics ===")
    for k, v in risk.items():
        print(f"{k}: {v}")



