from encoder import Encoder
from decoder import Decoder
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeVAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim, beta=1.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.beta = beta

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)
        return mu + eps * std
    
    def forward(self, x, c, y=None):
        ## x: (B, L, D)
        ## y: (B, H, D)

        #encode
        mu, logvar = self.encoder(x, c)

        #latent z
        z = self.reparameterize(mu, logvar)

        #decode
        mean, dist = self.decoder(z, c)

        #추론할때
        if y is None:
            return mean, z, (mu, logvar)

        #reconstruction loss
        recon_loss = -dist.log_prob(y).mean()

        #KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)

        #final loss
        loss = recon_loss + self.beta * kl_loss

        return loss, recon_loss, kl_loss, mean, z