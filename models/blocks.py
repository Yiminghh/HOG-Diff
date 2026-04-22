"""Common layers for defining score networks."""

import torch.nn as nn
import torch
import math


def get_act(config):
    """Get actiuvation functions from the config file."""

    if config.model.nonlinearity.lower() == 'elu':
        return nn.ELU()
    elif config.model.nonlinearity.lower() == 'relu':
        return nn.ReLU()
    elif config.model.nonlinearity.lower() == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif config.model.nonlinearity.lower() == 'swish':
        return nn.SiLU()
    elif config.model.nonlinearity.lower() == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError('activation function does not exist!')


def conv1x1(in_planes, out_planes, stride=1, bias=True, dilation=1, padding=0):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias, dilation=dilation,
                     padding=padding)
    return conv



class SinusoidalPosEmb(torch.nn.Module):
    # sinusoidal positional embeddings
    def __init__(self, dim, max_positions=10000):
        super().__init__()
        # magic number 10000 is from transformers
        self.dim = dim
        self.max_positions = max_positions

    def forward(self, x):
        device = x.device
        half_dim = (self.dim+1) // 2 # Adjust half_dim to handle odd dimensions
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb) #(5,)
        emb = x[:, None] * emb[None, :] #(80,1,5)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        emb = emb[:, :self.dim]  # Ensure the final dimension matches self.dim
        return emb

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        """SE-style channel gating on (B, N, nf) tensors; output shape matches input."""
        y = x.mean(dim=1)  # Global Average Pooling over nodes -> (B, nf).
        y = self.fc(y).unsqueeze(1)  # Learn per-channel weights -> (B, 1, nf).
        return x * y



class PNA(nn.Module):
    def __init__(self, d, dy):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier initialization.
        nn.init.xavier_uniform_(self.lin.weight)
        if self.lin.bias is not None:
            nn.init.zeros_(self.lin.bias)

    def forward(self, H):
        """ E: bs, n, n, de
            Features relative to the diagonal of E could potentially be added.
        """

        if H.dim() == 3:
            """ X: bs, n, dx. """
            m = H.mean(dim=1)
            mi = H.min(dim=1)[0]
            ma = H.max(dim=1)[0]
            std = H.std(dim=1)
        elif H.dim() == 4:
            """ E: bs, n, n, de. """
            m = H.mean(dim=(1, 2))
            mi = H.min(dim=2)[0].min(dim=1)[0]
            ma = H.max(dim=2)[0].max(dim=1)[0]
            std = torch.std(H, dim=(1, 2))
        else:
            raise NotImplementedError

        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out

class FilM(nn.Module):
    def __init__(self, nf, dropout):
        super().__init__()
        # FiLM y to h
        self.yh_add = nn.Linear(nf, nf)
        self.yh_mul = nn.Linear(nf, nf)

    def forward(self, h, y):
        # h: bs,n,nf
        # y: bs,nf

        yh1 = self.yh_add(y).unsqueeze(1) #(bs,1,nf)
        yh2 = self.yh_mul(y).unsqueeze(1)
        new_h = yh1 + (yh2 + 1) * h
        return new_h
