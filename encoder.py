import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
###############################################
#https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
###############################################

class ConditionLayer(nn.Module):
    def __init__(self, c_dim, h_dim):
        super().__init__()

        self.fc1 = nn.Linear(c_dim, h_dim)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)

    def forward(self, c):
        x = self.fc1(c)
        x = self.act(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs,
                 kernel_size, dilation, dropout=0.2):

        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                padding=padding,
                dilation=dilation
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                padding=padding,
                dilation=dilation
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1)
            if n_inputs != n_outputs
            else None
        )
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels,
                 kernel_size=3, dropout=0.2):

        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]

            layers.append(
                TemporalBlock(
                    n_inputs=in_ch,
                    n_outputs=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Encoder(nn.Module):
    def __init__(self, x_dim, c_dim, h_dim, z_dim,
                 tcn_layers=4, kernel_size=3):
        super().__init__()

        # Condition Network
        self.condition_layer = ConditionLayer(c_dim, h_dim)

        # Input projection (feature → hidden)
        self.input_proj = nn.Conv1d(x_dim, h_dim, kernel_size=1)

        # Vanilla Causal TCN
        self.tcn = TemporalConvNet(
            num_inputs=h_dim,
            num_channels=[h_dim] * tcn_layers,
            kernel_size=kernel_size,
            dropout=0.2
        )

        # Latent
        self.mu_layer = nn.Linear(h_dim, z_dim)
        self.logvar_layer = nn.Linear(h_dim, z_dim)

    def forward(self, x, c):
        """
        x: (B, L, x_dim)
        c: (B, c_dim)
        """

        # (B, x_dim, L)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)

        # (B, L, h_dim)
        x = x.permute(0, 2, 1)
        c_embed = self.condition_layer(c).unsqueeze(1)
        x = x + c_embed  # broadcast add

        # (B, h_dim, L)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)

        # Last timestep representation
        h_last = x[:, :, -1]

        mu = self.mu_layer(h_last)
        logvar = self.logvar_layer(h_last)

        return mu, logvar