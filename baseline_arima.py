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
# ARIMA Baseline (Rolling-forward)
# -------------------------------
def arima_baseline(X, Y, target_index=0, L=36, H=12):
    """
    X: (N, L, D)
    Y: (N, H, D)
    """
    N = X.shape[0]

    # 전체 target 시계열 만들기
    full_series = np.concatenate([
        X[:, -1, target_index],
        Y[-1, :, target_index]
    ])

    preds_all = []
    rmse_list = []
    nll_list = []
    crps_list = []

    for i in range(N):
        train_data = full_series[:(i + L)]

        model = ARIMA(train_data, order=(1,1,1)).fit()

        # σ: residual std
        sigma = np.std(model.resid) + 1e-6

        pred = model.forecast(H)
        preds_all.append(pred)

        y_true = Y[i, :, target_index]

        # RMSE
        rmse_list.append(
            np.sqrt(np.mean((y_true - pred)**2))
        )

        # NLL
        nll_list.append(
            np.mean(nll_gaussian(y_true, pred, sigma))
        )

        # CRPS
        crps_list.append(
            np.mean(crps_gaussian(y_true, pred, sigma))
        )

    return {
        "preds": np.array(preds_all),  # (N, H)
        "RMSE": float(np.mean(rmse_list)),
        "NLL": float(np.mean(nll_list)),
        "CRPS": float(np.mean(crps_list)),
    }
