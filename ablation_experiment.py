# ===========================================
# ablation_experiment.py
# Evaluate Ablation Models:
# - Posterior MSE / MAE / RMSE
# - NLL (reconstruction loss)
# - CRPS
# - Coverage / Sharpness (80% interval)
# - Risk Metrics (P_up, P_tail, VaR, ES)
# ===========================================

import torch
import numpy as np
import pandas as pd

from model import Encoder, Decoder, ConditionalPrior, TimeVAE
from macro_pretrain import MacroEncoder
from data_train import preprocess
from train_ablation_models import (
    EncoderNoFiLM, DecoderGaussian, DecoderNoDecomp,
    PriorNoMacro, TimeVAENoMacroPrior
)


# -------------------------------
# Metric Utils
# -------------------------------

def compute_point_forecast_metrics(preds, trues):
    preds = np.array(preds)
    trues = np.array(trues)

    diff = preds - trues
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))

    return {"MSE": mse, "MAE": mae, "RMSE": rmse}


def compute_coverage_and_sharpness(
    scenario_samples,
    true_future,
    feature_index=0,
    lower_q=10,
    upper_q=90
):
    scenario_samples = np.array(scenario_samples)
    true_future = np.array(true_future)

    lower = np.percentile(scenario_samples[:, :, feature_index], lower_q, axis=0)
    upper = np.percentile(scenario_samples[:, :, feature_index], upper_q, axis=0)
    true = true_future[:, feature_index]

    inside = (true >= lower) & (true <= upper)
    coverage = float(inside.mean())

    width = upper - lower
    sharpness = float(width.mean())

    return {
        "coverage": coverage,
        "sharpness": sharpness
    }


def compute_crps_from_samples(scenario_samples, true_future, feature_index=0):
    S = np.array(scenario_samples)[:, :, feature_index]  # (M,H)
    y = np.array(true_future)[:, feature_index]          # (H,)

    M, H = S.shape

    term1 = np.mean(np.abs(S - y[None, :]), axis=0)      # (H,)

    S1 = S[:, None, :]
    S2 = S[None, :, :]
    term2 = np.mean(np.abs(S1 - S2), axis=(0, 1))        # (H,)

    crps_per_h = term1 - 0.5 * term2
    crps = float(crps_per_h.mean())

    return {
        "CRPS_mean": crps,
        "CRPS_per_h": crps_per_h
    }


def compute_risk_metrics(
    scenario_samples,
    current_level_scaled,
    scaler,
    feature_index=0,
    horizon_idx=-1,
    tail_threshold_raw=-0.10,
    alpha=0.10
):
    scenario_samples = np.array(scenario_samples)
    M, H, D = scenario_samples.shape

    future_scaled = scenario_samples[:, horizon_idx, :]  # (M,D)
    future_raw = scaler.inverse_transform(future_scaled)[:, feature_index]

    current_raw = scaler.inverse_transform(
        np.array(current_level_scaled).reshape(1, -1)
    )[0, feature_index]

    ret = (future_raw - current_raw) / current_raw

    p_up = float(np.mean(ret > 0.0))
    p_tail = float(np.mean(ret < tail_threshold_raw))

    var_alpha = float(np.quantile(ret, alpha))
    tail = ret[ret <= var_alpha]
    if len(tail) < 3:
        es_alpha = var_alpha
    else:
        es_alpha = float(tail.mean())

    return {
        "P_up": p_up,
        "P_tail": p_tail,
        "VaR_10%": var_alpha,
        "ES_10%": es_alpha
    }


# -------------------------------
# Model Builder (eval용, train과 동일)
# -------------------------------

def build_model_for_name(name,
                         out_dim, cond_dim,
                         latent_dim, hidden, H,
                         macro_latent_dim,
                         macro_feature_indices,
                         device):

    if name == "full_ct_vae":
        encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
        decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H).to(device)
        prior   = ConditionalPrior(cond_dim, macro_latent_dim, latent_dim, hidden).to(device)

        macro_encoder = MacroEncoder(
            input_dim=6,
            hidden_dim=128,
            latent_dim=macro_latent_dim
        ).to(device)
        macro_encoder.load_state_dict(torch.load("macro_encoder.pth", map_location=device))
        macro_encoder.eval()
        for p in macro_encoder.parameters():
            p.requires_grad = False

        model = TimeVAE(
            encoder=encoder,
            decoder=decoder,
            prior=prior,
            macro_encoder=macro_encoder,
            latent_dim=latent_dim,
            beta=1.0
        ).to(device)
        ckpt_path = "full_ct_vae.pth"

    elif name == "no_student_t":
        encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
        decoder = DecoderGaussian(latent_dim, cond_dim, out_dim, hidden, H).to(device)
        prior   = ConditionalPrior(cond_dim, macro_latent_dim, latent_dim, hidden).to(device)

        macro_encoder = MacroEncoder(
            input_dim=6,
            hidden_dim=128,
            latent_dim=macro_latent_dim
        ).to(device)
        macro_encoder.load_state_dict(torch.load("macro_encoder.pth", map_location=device))
        macro_encoder.eval()
        for p in macro_encoder.parameters():
            p.requires_grad = False

        model = TimeVAE(
            encoder=encoder,
            decoder=decoder,
            prior=prior,
            macro_encoder=macro_encoder,
            latent_dim=latent_dim,
            beta=1.0
        ).to(device)
        ckpt_path = "no_student_t.pth"

    elif name == "no_macro_prior":
        encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
        decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H).to(device)
        prior   = PriorNoMacro(cond_dim, latent_dim, hidden).to(device)

        model = TimeVAENoMacroPrior(
            encoder=encoder,
            decoder=decoder,
            prior=prior,
            latent_dim=latent_dim,
            beta=1.0
        ).to(device)
        ckpt_path = "no_macro_prior.pth"

    elif name == "no_film":
        encoder = EncoderNoFiLM(out_dim, cond_dim, hidden, latent_dim).to(device)
        decoder = Decoder(latent_dim, cond_dim, out_dim, hidden, H).to(device)
        prior   = ConditionalPrior(cond_dim, macro_latent_dim, latent_dim, hidden).to(device)

        macro_encoder = MacroEncoder(
            input_dim=6,
            hidden_dim=128,
            latent_dim=macro_latent_dim
        ).to(device)
        macro_encoder.load_state_dict(torch.load("macro_encoder.pth", map_location=device))
        macro_encoder.eval()
        for p in macro_encoder.parameters():
            p.requires_grad = False

        model = TimeVAE(
            encoder=encoder,
            decoder=decoder,
            prior=prior,
            macro_encoder=macro_encoder,
            latent_dim=latent_dim,
            beta=1.0
        ).to(device)
        ckpt_path = "no_film.pth"

    elif name == "no_decomp":
        encoder = Encoder(out_dim, cond_dim, hidden, latent_dim).to(device)
        decoder = DecoderNoDecomp(latent_dim, cond_dim, out_dim, hidden, H).to(device)
        prior   = ConditionalPrior(cond_dim, macro_latent_dim, latent_dim, hidden).to(device)

        macro_encoder = MacroEncoder(
            input_dim=6,
            hidden_dim=128,
            latent_dim=macro_latent_dim
        ).to(device)
        macro_encoder.load_state_dict(torch.load("macro_encoder.pth", map_location=device))
        macro_encoder.eval()
        for p in macro_encoder.parameters():
            p.requires_grad = False

        model = TimeVAE(
            encoder=encoder,
            decoder=decoder,
            prior=prior,
            macro_encoder=macro_encoder,
            latent_dim=latent_dim,
            beta=1.0
        ).to(device)
        ckpt_path = "no_decomp.pth"

    else:
        raise ValueError(f"Unknown ablation name: {name}")

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# -------------------------------
# Single Ablation Evaluation
# -------------------------------

def evaluate_ablation_model(
    name,
    X, Y, C,
    macro_feature_indices,
    scaler,
    latent_dim, cond_dim, hidden, H,
    macro_latent_dim,
    device="cuda"
):
    print(f"\n===== Evaluating Ablation: {name} =====")

    out_dim = X.shape[-1]
    model = build_model_for_name(
        name,
        out_dim=out_dim,
        cond_dim=cond_dim,
        latent_dim=latent_dim,
        hidden=hidden,
        H=H,
        macro_latent_dim=macro_latent_dim,
        macro_feature_indices=macro_feature_indices,
        device=device
    )

    preds_list = []
    trues_list = []
    nll_list = []

    with torch.no_grad():
        for i in range(len(X)):
            x = torch.tensor(X[i:i+1]).float().to(device)
            y = torch.tensor(Y[i:i+1]).float().to(device)
            c = torch.tensor(C[i:i+1]).float().to(device)

            if model.macro_encoder is not None:
                macro_x = x[:, :, macro_feature_indices].permute(0, 2, 1)
            else:
                macro_x = None

            loss, recon, kl, mean, _, _ = model(
                x, c, macro_x,
                y=y,
                use_prior_sampling_if_no_y=False
            )

            preds_list.append(mean.cpu().numpy())
            trues_list.append(y.cpu().numpy())
            nll_list.append(recon.item())

    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)

    point_metrics = compute_point_forecast_metrics(preds, trues)
    nll_mean = float(np.mean(nll_list))

    # -------- Scenario Sampling (posterior 주변) --------
    with torch.no_grad():
        x_last = torch.tensor(X[-1:]).float().to(device)
        y_last = torch.tensor(Y[-1:]).float().to(device)
        c_last = torch.tensor(C[-1:]).float().to(device)

        if model.macro_encoder is not None:
            macro_x_last = x_last[:, :, macro_feature_indices].permute(0, 2, 1)
        else:
            macro_x_last = None

        # posterior q(z|x,c)
        mu_q, logvar_q = model.encoder(x_last, c_last)
        std_q = torch.exp(0.5 * logvar_q)

        num_samples = 50
        samples = []

        for _ in range(num_samples):
            eps = torch.randn_like(std_q)
            z = mu_q + std_q * eps
            mean_future, dist_future = model.decoder(z, c_last)
            y_scen = dist_future.rsample()
            samples.append(y_scen.squeeze(0).cpu().numpy())

    scenario_samples = np.stack(samples, axis=0)   # (M,H,D)

    cov = compute_coverage_and_sharpness(
        scenario_samples,
        true_future=Y[-1],
        feature_index=0,
        lower_q=10,
        upper_q=90
    )

    crps = compute_crps_from_samples(
        scenario_samples,
        true_future=Y[-1],
        feature_index=0
    )

    risk = compute_risk_metrics(
        scenario_samples=scenario_samples,
        current_level_scaled=X[-1][-1],
        scaler=scaler,
        feature_index=0,
        horizon_idx=-1,
        tail_threshold_raw=-0.10,
        alpha=0.10
    )

    return {
        "MSE": point_metrics["MSE"],
        "MAE": point_metrics["MAE"],
        "RMSE": point_metrics["RMSE"],
        "NLL": nll_mean,
        "CRPS": crps["CRPS_mean"],
        "Coverage80": cov["coverage"],
        "Sharpness": cov["sharpness"],
        "P_up": risk["P_up"],
        "P_tail": risk["P_tail"],
        "VaR_10%": risk["VaR_10%"],
        "ES_10%": risk["ES_10%"],
    }


# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    L = 36
    H = 12
    LATENT_DIM = 32
    HIDDEN = 128
    MACRO_LATENT_DIM = 32

    condition_raw_cols = ["Exchange Rate", "CAPEX", "PMI", "CLI", "ISM"]
    macro_cols = ["PMI", "GS10", "M2SL", "UNRATE", "CPIAUCSL", "INDPRO"]

    X, Y, C, scaler, df_raw, df_scaled, macro_feature_indices = preprocess(
        csv_path="data.csv",
        macro_csv_path="macro.csv",
        condition_raw_cols=condition_raw_cols,
        macro_cols=macro_cols,
        L=L,
        H=H
    )

    cond_dim = C.shape[-1]

    model_names = [
        "full_ct_vae",
        "no_student_t",
        "no_macro_prior",
        "no_film",
        "no_decomp",
    ]

    results = {}
    for name in model_names:
        results[name] = evaluate_ablation_model(
            name=name,
            X=X, Y=Y, C=C,
            macro_feature_indices=macro_feature_indices,
            scaler=scaler,
            latent_dim=LATENT_DIM,
            cond_dim=cond_dim,
            hidden=HIDDEN,
            H=H,
            macro_latent_dim=MACRO_LATENT_DIM,
            device=device
        )

    df_res = pd.DataFrame(results).T
    print("\n\n===== Final Ablation Summary =====")
    print(df_res.to_string(float_format=lambda x: f"{x: .4f}"))