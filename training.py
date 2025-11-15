import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from encoder import Encoder
from decoder import Decoder
from time_vae import TimeVAE

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
    cond_dim=10,
    out_dim=8,
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
    encoder = Encoder(latent_dim=latent_dim, cond_dim=cond_dim).to(device)
    decoder = Decoder(latent_dim=latent_dim, cond_dim=cond_dim, 
                      out_dim=out_dim, hidden=hidden, H=H).to(device)

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