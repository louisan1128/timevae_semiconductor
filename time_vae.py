import decoder
import encoder
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
        ##x: (B, L, D)    = input sequence (과거 36개월)
        ##y: (B, H, D)    = target sequence (미래 12개월)

        mu, logvar = self.encoder(x, c)

        z = self.reparameterize(mu, logvar)

        mean, dist = self.decoder(z, c)

        recon_loss = -dist.log_prob(y).mean()

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)

        loss = recon_loss + self.beta * kl_loss

        return loss, recon_loss, kl_loss, mean, z