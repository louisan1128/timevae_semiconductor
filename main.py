# ===========================================
# main.py — 전체 파이프라인 실행 (UPDATED)
# ===========================================

import numpy as np

from data_train_utils import (
    preprocess,
    train_model,
    rolling_backtest,
    rolling_forward_test,
)

from scenario_eval_utils import (
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
    # compute_risk_metrics,  # (원하면 나중에 MACRO_X 버전으로 같이 업데이트)
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

# 매크로 전처리(스케일러/변환)와 동일한 feature set을 쓴다는 전제
# (train preprocess 내부에서 macro_scaler.pkl 기반 변환/컬럼을 맞추도록 되어있어야 함)
MACRO_FEAT_COLS = [
    "PMI",
    "GS10",
    "LOG_M2",
    "UNRATE",
    "INFLATION",
    "LOG_INDPRO",
]

# ===========================================
# Main
# ===========================================
if __name__ == "__main__":

    print("========== 1) Preprocessing ==========")

    # preprocess() UPDATED:
    # - X: (N, L, D)
    # - Y: (N, H, D)
    # - C: (N, cond_dim)
    # - MACRO_X: (N, 6, L)  <-- 중요!
    X, Y, C, MACRO_X, scaler, df_raw, df_scaled, macro_feat_cols = preprocess(
        csv_path="data.csv",
        macro_csv_path="macro.csv",
        condition_raw_cols=condition_raw_cols,
        macro_feat_cols=MACRO_FEAT_COLS,
        L=L,
        H=H,
    )

    print("X:", X.shape, "Y:", Y.shape, "C:", C.shape, "MACRO_X:", MACRO_X.shape)
    print("macro_feat_cols:", macro_feat_cols)
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
    )

    print("=================================\n")

    # =======================================
    # 3) Rolling Backtest (optional)
    # =======================================
    # print("========== 3) Rolling Backtest ==========")
    # mse_back = rolling_backtest(
    #     model_path="timevae_ctvae_prior.pth",
    #     X=X, Y=Y, C=C, MACRO_X=MACRO_X,
    #     latent_dim=LATENT_DIM,
    #     cond_dim=COND_DIM,
    #     hidden=HIDDEN,
    #     H=H,
    #     beta=BETA,
    #     macro_hidden_dim=MACRO_HIDDEN_DIM,
    #     macro_latent_dim=MACRO_LATENT_DIM,
    #     device=DEVICE,
    # )
    # print(f"Rolling Backtest MSE: {mse_back:.6f}")
    # print("=================================\n")

    # =======================================
    # 4) Rolling Forward Test (optional)
    # =======================================
    # print("========== 4) Rolling Forward Test ==========")
    # mse_forward = rolling_forward_test(
    #     X=X, Y=Y, C=C, MACRO_X=MACRO_X,
    #     latent_dim=LATENT_DIM,
    #     cond_dim=COND_DIM,
    #     hidden=HIDDEN,
    #     H=H,
    #     beta=BETA,
    #     lr=LR,
    #     epochs=10,
    #     batch_size=BATCH_SIZE,
    #     macro_hidden_dim=MACRO_HIDDEN_DIM,
    #     macro_latent_dim=MACRO_LATENT_DIM,
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

    # 마지막 관측 RAW 값 (condition/feature 편집에 사용)
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

    # raw → scaled 변환(전체 feature 벡터 만들기)
    full_raw_vec = [
        scenario_cond_raw[col] if col in scenario_cond_raw else last_truth_raw[col]
        for col in df_raw.columns
    ]
    scaled_full = scaler.transform([full_raw_vec])[0]

    # condition만 추출
    scenario_cond_scaled = np.array([
        scaled_full[df_raw.columns.get_loc(col)]
        for col in condition_raw_cols
    ], dtype=np.float32)

    # === scenario sampling ===
    scenario_samples = scenario_predict_local(
        model_path="timevae_ctvae_prior.pth",
        X_last=X[-1],                      # (L,D)
        macro_x_last=MACRO_X[-1],          # (6,L)  <-- 중요!
        cond_true=C[-1],                   # (cond_dim,)
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
        sample_from_student_t=True,
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
        model_path="timevae_ctvae_prior.pth",
        X=X,
        C=C,
        MACRO_X=MACRO_X,
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        hidden=HIDDEN,
        H=H,
        beta=BETA,
        macro_hidden_dim=MACRO_HIDDEN_DIM,
        macro_latent_dim=MACRO_LATENT_DIM,
        device=DEVICE
    )

    # 2) 마지막 시점 posterior scenario
    samples_post = posterior_scenario(
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

    # true 전체 시계열 (각 chunk의 첫 y가 true future 1-step)
    true_full = Y[:, 0, :]

    plot_full_forecast_and_scenario(
        true_full=true_full,
        forecast_full=forecast_full,
        scenario_samples=scenario_samples,
        feature_index=0,
        H=H
    )

    print("=========== Completed 2nd! ===========")

    # =======================================
    # 8) Metrics
    # =======================================
    point_metrics = compute_point_forecast_metrics(preds, trues)
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
