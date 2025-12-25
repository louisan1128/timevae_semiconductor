# baseline_arima.py
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm

# -------------------------------
# Gaussian NLL
# -------------------------------
def nll_gaussian(y, mu, sigma):
    eps = 1e-6
    return 0.5 * np.log(2*np.pi*(sigma**2) + eps) + \
           0.5 * ((y - mu)**2) / (sigma**2 + eps)

# -------------------------------
# Gaussian CRPS
# -------------------------------
def crps_gaussian(y, mu, sigma):
    z = (y - mu) / sigma
    return sigma * (z * (2*norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1/np.sqrt(np.pi))

# -------------------------------
# Coverage / Sharpness
# -------------------------------
def gaussian_coverage_and_sharpness(y_true, mu, sigma, lower_q=10, upper_q=90):
    """
    y_true: (H,)
    mu    : (H,)
    sigma : scalar
    """
    z_low = norm.ppf(lower_q / 100.0)
    z_up  = norm.ppf(upper_q / 100.0)

    lower = mu + sigma * z_low
    upper = mu + sigma * z_up

    inside = (y_true >= lower) & (y_true <= upper)
    coverage = inside.mean()

    sharpness = (upper - lower).mean()

    return coverage, sharpness

# -------------------------------
# ARIMA Baseline (Rolling-forward)
# -------------------------------
def arima_baseline(X, Y, target_index=0, L=36, H=12):
    """
    X: (N, L, D)
    Y: (N, H, D)
    """
    N = X.shape[0]

    # 전체 target 시계열
    full_series = np.concatenate([
        X[:, -1, target_index],
        Y[-1, :, target_index]
    ])

    preds_all = []
    rmse_list = []
    nll_list = []
    crps_list = []

    coverage_list = []
    sharpness_list = []

    for i in range(N):
        train_data = full_series[:(i + L)]

        model = ARIMA(train_data, order=(1,1,1)).fit()

        sigma = np.std(model.resid) + 1e-6   # residual std

        pred = model.forecast(H)             # (H,)
        preds_all.append(pred)

        y_true = Y[i, :, target_index]

        # RMSE
        rmse_list.append(np.sqrt(np.mean((y_true - pred)**2)))

        # NLL
        nll_list.append(np.mean(nll_gaussian(y_true, pred, sigma)))

        # CRPS
        crps_list.append(np.mean(crps_gaussian(y_true, pred, sigma)))

        # Coverage + Sharpness
        cov_i, sharp_i = gaussian_coverage_and_sharpness(
            y_true=y_true,
            mu=pred,
            sigma=sigma,
            lower_q=10,
            upper_q=90
        )
        coverage_list.append(cov_i)
        sharpness_list.append(sharp_i)

    return {
        "preds": np.array(preds_all),      # (N, H)
        "RMSE": float(np.mean(rmse_list)),
        "NLL": float(np.mean(nll_list)),
        "CRPS": float(np.mean(crps_list)),
        "Coverage_80%": float(np.mean(coverage_list)),
        "Sharpness_80%": float(np.mean(sharpness_list)),
    }
