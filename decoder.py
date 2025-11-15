import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import StudentT
import math

latent_dim = 32
condition_dim = 10
B = 32  ##batch size
L = 36  ##sequnce length
D = 8   ##feature dimension
H = 12  ##predicting length
hidden = 128 ##hidden layer


class Decoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, out_dim, hidden, H):
        super().__init__()
        self.H = H
        self.out_dim = out_dim

        self.fc_context = nn.Linear(latent_dim + cond_dim, hidden)

        # Trend
        self.trend = nn.Linear(hidden, H * out_dim)

        # Seasonality
        self.season = nn.Linear(hidden, H * out_dim)

        # Residual RNN 
        self.rnn = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            batch_first=True
        )
        self.rnn_out = nn.Linear(hidden, out_dim)

        # Student-t parameters
        self.fc_scale = nn.Linear(hidden, H * out_dim)
        self.fc_df    = nn.Linear(hidden, 1)


    def forward(self, z, c):
        x = torch.cat([z, c], dim=-1)  # (B, latent_dim + cond_dim)
        h = torch.relu(self.fc_context(x)) # (B, hidden)

        ## Trend
        trend = self.trend(h).view(-1, self.H, self.out_dim)

        ## Seasonality
        sea = torch.sin(self.season(h)).view(-1, self.H, self.out_dim)

        ## Residual RNN
        rnn_input = h.unsqueeze(1).repeat(1, self.H, 1)   # (B, H, hidden)
        rnn_out, _ = self.rnn(rnn_input)                  # (B, H, hidden)
        residual = self.rnn_out(rnn_out)                  # (B, H, out_dim)

        ## Final mean
        mean = trend + sea + residual

        ## Student-t params
        scale = torch.softplus(self.fc_scale(h)).view(-1, self.H, self.out_dim) + 1e-3
        df    = torch.softplus(self.fc_df(h)) + 2.0   # degree of freedom > 2

        dist = StudentT(df, loc=mean, scale=scale)

        return mean, dist