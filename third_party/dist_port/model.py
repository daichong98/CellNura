"""DIST core port from DRFNS UNetDistance/UNetBatchNorm/UNetObject.
Original source remains in ../source. This is a reimplementation, not author weights.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TFBatchNorm(nn.Module):
    """TF author's moments + EMA: population variance, zero-initialized shadows.
    torch BatchNorm's running variance update is different; implement explicitly.
    """
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(channels).normal_(1., .02))
        self.beta = nn.Parameter(torch.zeros(channels))
        self.register_buffer("moving_mean", torch.zeros(channels))
        self.register_buffer("moving_var", torch.zeros(channels))

    def forward(self, x):
        if self.training:
            var, mean = torch.var_mean(x, dim=(0, 2, 3), unbiased=False)
            with torch.no_grad():
                self.moving_mean.mul_(.9).add_(mean.detach(), alpha=.1)
                self.moving_var.mul_(.9).add_(var.detach(), alpha=.1)
        else:
            mean, var = self.moving_mean, self.moving_var
        shape = (1, -1, 1, 1)
        return (x - mean.view(shape)) * torch.rsqrt(var.view(shape) + 1e-5) * self.gamma.view(shape) + self.beta.view(shape)


def init_kernel(layer, fan_in):
    # Author's weight_xavier is truncated normal std=sqrt(1/fan_in), NOT Glorot.
    std = math.sqrt(1. / fan_in)
    nn.init.trunc_normal_(layer.weight, std=std, a=-2*std, b=2*std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, .1)


class ConvBNBiasReLU(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=0, bias=False)
        init_kernel(self.conv, 9 * cin)
        self.bn = TFBatchNorm(cout)
        self.bias = nn.Parameter(torch.full((cout,), .1))

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)) + self.bias[None, :, None, None])


class DIST(nn.Module):
    """F16 VALID U-Net, one linear distance output; no invented type head/dropout."""
    def __init__(self, features=16):
        super().__init__()
        channels = [features * 2**i for i in range(5)]
        self.enc = nn.ModuleList()
        cin = 3
        for cout in channels:
            self.enc.append(nn.Sequential(ConvBNBiasReLU(cin, cout), ConvBNBiasReLU(cout, cout)))
            cin = cout
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for cout in reversed(channels[:-1]):
            up = nn.ConvTranspose2d(2*cout, cout, 2, stride=2)
            # TF transpose filter [2,2,out,in]; author's initialization uses out.
            init_kernel(up, 4*cout)
            self.up.append(up)
            self.dec.append(nn.Sequential(ConvBNBiasReLU(2*cout, cout), ConvBNBiasReLU(cout, cout)))
        self.head = nn.Conv2d(features, 1, 1)
        init_kernel(self.head, features)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Expected NCHW RGB")
        if any(s < 188 or (s - 12) % 16 for s in x.shape[-2:]):
            raise ValueError("Use spatial sizes 16*k+12 >=188 (396 -> 212, 572 -> 388)")
        skips = []
        for i, block in enumerate(self.enc):
            if i:
                x = F.max_pool2d(x, 2, 2)
            x = block(x)
            skips.append(x)
        for up, block, skip in zip(self.up, self.dec, reversed(skips[:-1])):
            x = F.relu(up(x))
            dy, dx = (skip.shape[-2]-x.shape[-2])//2, (skip.shape[-1]-x.shape[-1])//2
            skip = skip[:, :, dy:dy+x.shape[-2], dx:dx+x.shape[-1]]
            x = block(torch.cat((skip, x), dim=1))
        return self.head(x)
