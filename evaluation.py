import torch
import numpy as np

from encoder import Encoder
from decoder import Decoder
from time_vae import TimeVAE


def evaluate_model(
    model_path,
    X,
    Y,
    C,
    latent_dim=32,
    cond_dim=10,
    hidden=128,
    out_dim=8,
    H=12,
    device="cuda"
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    encoder = Encoder(
        x_dim=X.shape[-1],
        c_dim=cond_dim,
        h_dim=hidden,
        z_dim=latent_dim
    ).to(device)

    decoder = Decoder(
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        out_dim=out_dim,
        hidden=hidden,
        H=H
    ).to(device)

    model = TimeVAE(
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        beta=1.0
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds = []
    trues = []

    with torch.no_grad():
        for i in range(len(X)):
            x = torch.tensor(X[i:i+1], dtype=torch.float32).to(device)
            c = torch.tensor(C[i:i+1], dtype=torch.float32).to(device)
            y = torch.tensor(Y[i:i+1], dtype=torch.float32).to(device)

            # inference mode → y=None
            pred_mean, z, _ = model(x, c, y=None)

            preds.append(pred_mean.cpu().numpy())
            trues.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)  
    trues = np.concatenate(trues, axis=0) 

    mse = np.mean((preds - trues) ** 2)
    print("===================================")
    print(f"Evaluation MSE: {mse:.6f}")
    print("===================================")

    return preds, trues, mse

