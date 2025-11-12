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


class Decoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, out_dim):
        super().__init__()

    def forward(self, z, c):
        return mean, dist