import torch
import torch.nn as nn 
import torch.nn.functional as F

import numpy as np 
import math 

from globals_var import * 

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=KERNEL_SIZE, padding=1)
        N = KERNEL_SIZE**2 * in_ch
        torch.nn.init.normal_(self.conv.weight, std=np.sqrt(2/N))    # from original U-Net paper
        self.nonLinear = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_ch)
        time_dim = 32
        self.time_emb = SinusoidalPositionEmbeddings(time_dim)
        self.time_transform = nn.Linear(time_dim, out_ch)

    def forward(self, x, t):
        h = self.norm(self.nonLinear(self.conv(x)))
        time_embedding = self.time_emb(t)
        time_embedding = self.time_transform(time_embedding)
        time_embedding = time_embedding[(..., ) + (None, ) * 2]
        return h + time_embedding


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, depth, scaling):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        conv_blocks = []
        for i in range(depth):
            if i == 0:
                in_channels = self.in_ch
            else:
                in_channels = self.out_ch
            conv_block = ConvBlock(in_channels, out_ch)
            conv_blocks.append(conv_block)
        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.scaling = scaling

    def forward(self, x, t):
        for conv_l in self.conv_blocks:
            x = conv_l(x, t)
        return x


class UNet(nn.Module):
    def __init__(self, ch, ch_mult, depth, attn_resolutions=None):
        super().__init__()
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_resolutions = len(ch_mult)
        self.depth = depth
        self.attn_resolutions = attn_resolutions
        self.in_ch = 3
        downs = []
        ups = []
        in_ch = self.in_ch
        for mult in ch_mult[:-1]:
            out_ch = self.ch*mult
            downs.append(Block(in_ch, out_ch, depth, nn.MaxPool2d(2)))
            in_ch = out_ch
        in_ch = ch
        for mult in ch_mult[1:]:
            out_ch = self.ch*mult
            ups.append(Block(out_ch, in_ch, depth, nn.ConvTranspose2d(out_ch, in_ch, kernel_size=2, stride=2)))
            in_ch = out_ch
        self.downs = nn.Sequential(*downs)
        middle_in = ch_mult[-2]*ch
        middle_max = ch_mult[-1]*ch
        self.middle = Block(middle_in, middle_max, depth, None)
        self.ups = nn.Sequential(*reversed(ups))
        self.last_conv = nn.Conv2d(ch, self.in_ch, 1)

    def forward(self, x, time):
        residual_connections = []
        for down in self.downs:
            x = down(x, time)
            residual_connections.append(x)
            x = down.scaling(x)
        x = self.middle(x, time)
        for up in self.ups:
            x = up.scaling(x)
            res = residual_connections.pop()
            x = torch.cat([x, res], dim=1)
            x = up(x, time)
        x = self.last_conv(x)
        return x

class ModelCombined(nn.Module):
    def __init__(self, model_content, model_fast, diffusion):
        super().__init__()
        self.model_content = model_content
        self.model_fast = model_fast
        self.diffusion = diffusion 

    def forward(self, x, time):
        """
        This method should be only used for sampling. It assumes that time is an uniform vector. 
        """
        t = time[0]
        snr = self.diffusion.snr[int(t)]
        if snr <= 0.01 or snr > 1:
            return self.model_fast(x, time)
        else:
            return self.model_content(x, time)

