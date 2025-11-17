import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from encoder import Encoder
from decoder import Decoder
from time_vae import TimeVAE
from preprocessing_2 import preprocess

###dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, c):
        """
        x: (N, L, D)
        y: (N, H, D)
        c: (N, cond_dim)
        """
        self.x = x
        self.y = y
        self.c = c

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.c[idx]


###training
def train_model(
    x_train, y_train, c_train,
    latent_dim=32,
    cond_dim=5,
    out_dim=7,
    hidden=128,
    H=12,
    lr=1e-3,
    epochs=100,
    batch_size=32,
    beta=1.0,
    device="cuda"
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Dataset
    train_dataset = TimeSeriesDataset(x_train, y_train, c_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Build model
    encoder = Encoder(x_dim=x_train.shape[-1],
                      c_dim=cond_dim,
                      h_dim=hidden,
                      z_dim=latent_dim).to(device)
    
    decoder = Decoder(latent_dim=latent_dim, 
                      cond_dim=cond_dim, 
                      out_dim=out_dim, 
                      hidden=hidden, H=H).to(device)

    model = TimeVAE(encoder, decoder, latent_dim=latent_dim, beta=beta).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print('========Training start========')
    for epoch in range(1, epochs + 1):

        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        num_batches = 0

        for x_batch, y_batch, c_batch in train_loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            c_batch = c_batch.to(device)

            # TimeVAE forward: loss, recon, kl, mean, z
            loss, recon, kl, mean, z = model(x_batch, c_batch, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()
            num_batches += 1

        print(f"[Epoch {epoch}/{epochs}] "
              f"Loss: {epoch_loss/num_batches:.4f} | "
              f"Recon: {epoch_recon/num_batches:.4f} | "
              f"KL: {epoch_kl/num_batches:.4f}")

    print("========== Training Finished ==========")

    torch.save(model.state_dict(), "timevae_model.pth")
    print("Saved model → timevae_model.pth")

    return model




####evaluation
def evaluate_model(
    model_path,
    X,
    Y,
    C,
    latent_dim=32,
    cond_dim=5,
    hidden=128,
    out_dim=7,
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


##custom condition
def scenario_predict(model_path, X_last, custom_condition,
                     latent_dim=32, cond_dim=5, hidden=128, out_dim=7, H=12,
                     device="cuda"):

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    encoder = Encoder(x_dim=X_last.shape[-1], c_dim=cond_dim,
                      h_dim=hidden, z_dim=latent_dim).to(device)
    decoder = Decoder(latent_dim=latent_dim, cond_dim=cond_dim,
                      out_dim=out_dim, hidden=hidden, H=H).to(device)

    model = TimeVAE(encoder, decoder, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    X_tensor = torch.tensor(X_last, dtype=torch.float32).unsqueeze(0).to(device)
    C_tensor = torch.tensor(custom_condition, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_mean, z, dist = model(X_tensor, C_tensor, y=None)

    return pred_mean.cpu().numpy()[0]



##시각화
def plot_forecast(true, pred, title="Forecast vs Real", feature_index=0):
    plt.figure(figsize=(8,4))
    plt.plot(true[:, feature_index], label="Real", color="black")
    plt.plot(pred[:, feature_index], label="Prediction", color="red")
    plt.title(title)
    plt.legend()
    plt.show()




if __name__ == "__main__":
    # 1) 전처리 실행
    X, Y, C = preprocess()

    # 2) 학습 실행
    train_model(X, Y, C)

    # 3) 평가 실행
    preds, trues, mse = evaluate_model("timevae_model.pth", X, Y, C)

    plot_forecast(trues[1], preds[1], title="Example Prediction")

    # ---------------- SCENARIO FORECAST ----------------
    # custom macro condition:
    # [Exchange, CAPEX, PMI, CLI, ISM]
    custom_condition = [1300, 5.0, 52.0, 101.0, 50.0]

    future_pred = scenario_predict(
        "timevae_model.pth",
        X_last=X[-1],         # 마지막 36개월 입력
        custom_condition=custom_condition
    )
    

    print("Scenario Forecast (12 months):")
    print(future_pred)