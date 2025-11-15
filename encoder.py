import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

latent_dim = 32
condition_dim = 10
B = 32  ##batch size
L = 36  ##sequnce length
D = 8   ##feature dimension
H = 12  ##predicting length

class ConditionLayer(nn.Module):
    def __init__(self, c_dim, h_dim):
        super().__init__()

        # 1st FC + ReLU + LayerNorm
        self.fc1 = nn.Linear(c_dim, h_dim)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(h_dim)

        # 2nd FC
        self.fc2 = nn.Linear(h_dim, h_dim)

    def forward(self, c):

        x = self.fc1(c)     # Linear(c_dim → h_dim)
        x = self.act(x)     # ReLU
        x = self.norm(x)    # LayerNorm(h_dim)
        x = self.fc2(x)     # Linear(h_dim → h_dim)

        return x

class MixtureTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilations, dropout=0.2):
        super().__init__()

        self.dilations = dilations
        self.num_branches = len(dilations)

        # conv1 branches
        self.conv1_branches = nn.ModuleList([
            weight_norm(nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                padding=(kernel_size - 1) * d,
                dilation=d
            ))
            for d in dilations
        ])

        # conv2 branches
        self.conv2_branches = nn.ModuleList([
            weight_norm(nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                padding=(kernel_size - 1) * d,
                dilation=d
            ))
            for d in dilations
        ])

        # learnable gating
        self.gate_logits1 = nn.Parameter(torch.zeros(self.num_branches))
        self.gate_logits2 = nn.Parameter(torch.zeros(self.num_branches))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) \
            if n_inputs != n_outputs else None

        self.init_weights()

    def init_weights(self):
        for conv in list(self.conv1_branches) + list(self.conv2_branches):
            conv.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        x: (B, C, L)
        """

        # ---- conv1 mixture ----
        gate1 = torch.softmax(self.gate_logits1, dim=0)
        conv1_outs = [w * conv(x) for w, conv in zip(gate1, self.conv1_branches)]
        out = sum(conv1_outs)
        out = self.relu(out)
        out = self.dropout(out)

        # ---- conv2 mixture ----
        gate2 = torch.softmax(self.gate_logits2, dim=0)
        conv2_outs = [w * conv(out) for w, conv in zip(gate2, self.conv2_branches)]
        out = sum(conv2_outs)
        out = self.relu(out)
        out = self.dropout(out)

        # ---- residual ----
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class MixtureTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels,
                 kernel_size=3, dilations=[1, 2, 4, 8], dropout=0.2):
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]

            layers.append(
                MixtureTemporalBlock(
                    n_inputs=in_ch,
                    n_outputs=out_ch,
                    kernel_size=kernel_size,
                    dilations=dilations,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Encoder(nn.Module):
    def __init__(self, x_dim, c_dim, h_dim, z_dim,
                 tcn_layers=4, kernel_size=3, dilations=[1,2,4,8]):
        super().__init__()

        # ---- Condition ----
        self.condition_layer = ConditionLayer(c_dim, h_dim)

        # ---- Input Projection ----
        self.input_proj = nn.Conv1d(x_dim, h_dim, kernel_size=1)

        # ---- Mixture TCN ----
        self.tcn = MixtureTemporalConvNet(
            num_inputs=h_dim,
            num_channels=[h_dim] * tcn_layers,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=0.2
        )

        # ---- Latent Projection ----
        self.mu_layer = nn.Linear(h_dim, z_dim)
        self.logvar_layer = nn.Linear(h_dim, z_dim)

    def forward(self, x, c):
        """
        x: (B, L, x_dim)
        c: (B, c_dim)
        """

        B, L, _ = x.shape

        # --- Input Projection ---
        x = x.permute(0, 2, 1)         # (B, x_dim, L)
        x = self.input_proj(x)         # (B, h_dim, L)

        # --- Condition Projection ---
        x = x.permute(0, 2, 1)         # (B, L, h_dim)
        c_embed = self.condition_layer(c).unsqueeze(1)    # (B, 1, h_dim)
        x = x + c_embed                # broadcast add

        # --- TCN ---
        x = x.permute(0, 2, 1)         # (B, h_dim, L)
        x = self.tcn(x)                # (B, h_dim, L)

        # --- Latent (last timestep) ---
        h_last = x[:, :, -1]           # (B, h_dim)
        mu = self.mu_layer(h_last)
        logvar = self.logvar_layer(h_last)

        return mu, logvar
