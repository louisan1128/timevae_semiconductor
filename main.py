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
EPOCHS = 40          # rolling forward 재학습 시 epoch 많으면 오래 걸림
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

    X, Y, C, scaler, df_raw = preprocess(
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

    # 시나리오 조건 (RAW 형태)
    scenario_cond_raw = {
        "Exchange Rate": 1500,
        "CAPEX": 0.0,
        "Global Manufacturing PMI": 53.0,
        "OECD CLI": 100.5,
        "U.S. ISM Manufacturing New Orders Index": 49.3,
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

    print("Scenario samples shape:", scenario_samples.shape)

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

    print("=========== Completed! ===========")
