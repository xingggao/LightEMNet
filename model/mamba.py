import math
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
# from model.MAMBA import Mamba
from mamba_ssm.modules.mamba_simple import Mamba
from model.mamba_back import VSSBlock

class SingleMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # self.norm1 = nn.LayerNorm(dim)
        # self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v5',
        #                     if_devide_out=True, use_norm=True, input_h=H, input_w=W)
        self.block = Mamba(dim, expand=2)
        # self.block = VSSBlock(dim)


    def forward(self, input):
        # input: (B, N, C)
        skip = input
        input = self.norm(input)
        output = self.block(input)
        # output = self.norm1(output)
        return output + skip


class CAMMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        # self.norm2 = nn.LayerNorm(dim)
        # self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v5',
        #                     if_devide_out=True, use_norm=True, input_h=H, input_w=W)
        # self.block = VSSBlock(dim)
        self.block = Mamba(dim, expand=1)


    def forward(self, input):
        skip = input
        # input = input.permute(0,2,3,1)
        # input = input.permute(0, 3, 1, 2)
        b, c, h, w = input.shape
        input = rearrange(input, 'b c h w -> b (h w) c', h=h, w=w)
        input = self.norm0(input)
        input = self.block(input)
        pan = rearrange(input, 'b (h w) c -> b c h w', h=h, w=w)
        return pan + skip

class DirMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        # self.norm2 = nn.LayerNorm(dim)
        # self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v5',
        #                     if_devide_out=True, use_norm=True, input_h=H, input_w=W)
        # self.block = VSSBlock(dim)
        self.block = Mamba(dim, expand=1)
        self.out_proj = nn.Linear(dim, dim // 4)

    def forward(self, input):
        skip = input
        # input = input.permute(0,2,3,1)
        # input = input.permute(0, 3, 1, 2)
        b, c, h, w = input.shape
        input = rearrange(input, 'b c h w -> b (h w) c', h=h, w=w)
        input = self.norm0(input)
        input = self.block(input)
        pan = rearrange(input, 'b (h w) c -> b c h w', h=h, w=w)
        pan = pan + skip
        pan = self.out_proj(pan)
        return pan
# class FusionMamba(nn.Module):
#     def __init__(self, dim, H, W, depth=1):
#         super().__init__()
#
#         self.spa_mamba_layers = SingleMambaBlock(dim)
#         self.spe_mamba_layers = SingleMambaBlock(dim)
#         self.spa_cross_mamba = CrossMambaBlock(dim, H, W)
#         self.spe_cross_mamba = CrossMambaBlock(dim, H, W)
#         self.out_proj = nn.Linear(dim, dim)
#
#     def forward(self, pan, ms):
#         b, c, h, w = pan.shape
#         pan = rearrange(pan, 'b c h w -> b (h w) c', h=h, w=w)
#         ms = rearrange(ms, 'b c h w -> b (h w) c', h=h, w=w)
#
#         pan = self.spa_mamba_layers(pan)
#         ms = self.spe_mamba_layers(ms)
#         spa_fusion = self.spa_cross_mamba(pan, ms)
#         spe_fusion = self.spe_cross_mamba(ms, pan)
#         fusion = self.out_proj((spa_fusion + spe_fusion) / 2)
#         pan = rearrange(pan, 'b (h w) c -> b c h w', h=h, w=w)
#         ms = rearrange(ms, 'b (h w) c -> b c h w', h=h, w=w)
#         output = rearrange(fusion, 'b (h w) c -> b c h w', h=h, w=w)
#         return pan, ms + output

class FusionMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.spa_mamba_layers = SingleMambaBlock(dim)
        self.spe_mamba_layers = SingleMambaBlock(dim)
        # self.spa_cross_mamba = CrossMambaBlock(dim)
        # self.spe_cross_mamba = CrossMambaBlock(dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, pan, ms):
        b, c, h, w = pan.shape
        b, y, z, a = ms.shape
        pan1 = rearrange(pan, 'b c h w -> b (h w) c', h=h, w=w)
        ms1 = rearrange(ms, 'b c h w -> b (h w) c', h=z, w=a)
        cat = torch.cat((pan1, ms1), dim=1)

        fusion = self.spa_mamba_layers(cat)

        # spa_fusion = self.spa_cross_mamba(pan, ms)
        # spe_fusion = self.spe_cross_mamba(ms, pan)
        # fusion = self.out_proj((spa_fusion + spe_fusion) / 2)
        pan, ms = torch.split(fusion, [h*w,z*a], dim=1)
        pan = rearrange(pan, 'b (h w) c -> b c h w', h=h, w=w)
        ms = rearrange(ms, 'b (h w) c -> b c h w', h=z, w=a)

        output = pan + ms
        # output = rearrange(fusion, 'b (h w) c -> b c h w', h=h, w=w)
        return output
        # return pan, ms + output

class FusionMamba5(nn.Module):
    def __init__(self, dim1):
        super().__init__()

        self.spa_mamba_layers = SingleMambaBlock(dim1)
        self.spe_mamba_layers = SingleMambaBlock(dim1)
        # self.spa_cross_mamba = CrossMambaBlock(dim)
        # self.spe_cross_mamba = CrossMambaBlock(dim)
        self.norm = nn.LayerNorm(dim1)
        self.out_proj = nn.Linear(dim1, dim1)
        self.skip_scale = nn.Parameter(torch.ones(1))


    def forward(self, pan, ms):
        b, c, h, w = pan.shape
        b, y, z, a = ms.shape
        pan1 = rearrange(pan, 'b c h w -> b (h w) c', h=h, w=w)
        ms1 = rearrange(ms, 'b c h w -> b (h w) c', h=z, w=a)
        cat = torch.cat((pan1, ms1), dim=1)

        fusion = self.spa_mamba_layers(cat)
        pan_f, ms_f = torch.split(fusion, [h * w, z * a], dim=1)



        # pan = self.spa_mamba_layers(pan1) + self.skip_scale * pan1
        # ms = self.spa_mamba_layers(ms1) + self.skip_scale * ms1
        pan = self.spa_mamba_layers(pan1) * pan_f
        ms = self.spa_mamba_layers(ms1) * ms_f
        pan = rearrange(pan, 'b (h w) c -> b c h w', h=h, w=w)
        ms = rearrange(ms, 'b (h w) c -> b c h w', h=z, w=a)
        output = pan + ms
        # output = rearrange(fusion, 'b (h w) c -> b c h w', h=h, w=w)
        return output

class FusionMamba55(nn.Module):
    def __init__(self, dim1):
        super().__init__()
        self.block = Mamba(dim1, expand=2)
        # self.spa_cross_mamba = CrossMambaBlock(dim)
        # self.spe_cross_mamba = CrossMambaBlock(dim)
        self.norm = nn.LayerNorm(dim1)
        self.out_proj = nn.Linear(dim1, dim1)
        self.skip_scale = nn.Parameter(torch.ones(1))


    def forward(self, pan, ms):
        b, c, h, w = pan.shape
        b, y, z, a = ms.shape
        pan1 = rearrange(pan, 'b c h w -> b (h w) c', h=h, w=w)
        ms1 = rearrange(ms, 'b c h w -> b (h w) c', h=z, w=a)
        cat = torch.cat((pan1, ms1), dim=1)
        cat_l = self.norm(cat)


        fusion = self.block(cat_l) + cat * self.skip_scale
        fusion = self.out_proj(self.norm(fusion))
        pan_f, ms_f = torch.split(fusion, [h * w, z * a], dim=1)
        # pan = self.spa_mamba_layers(pan1) + self.skip_scale * pan1
        # ms = self.spa_mamba_layers(ms1) + self.skip_scale * ms1
        # ms = self.spa_mamba_layers(ms1) * ms_f
        # pan = pan_f + self.skip_scale * ms
        pan = rearrange(pan_f, 'b (h w) c -> b c h w', h=h, w=w)
        ms = rearrange(ms_f, 'b (h w) c -> b c h w', h=z, w=a)
        output = pan + ms

        # output = rearrange(fusion, 'b (h w) c -> b c h w', h=h, w=w)
        return output
class MambaB(nn.Module):
    def __init__(self, dim,dim1):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim+dim1)
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim+dim1,  # Model dimension d_model
        )
        # self.proj = nn.Linear(dim, out_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.out_proj = nn.Linear(dim+dim1, dim)

    def forward(self, input,input2):
        # input0: (B, N, C) | input1: (B, N, C)
        b, c, h, w = input.shape
        input = torch.cat((input,input2),dim=1)
        pan = rearrange(input, 'b c h w -> b (h w) c', h=h, w=w)

        input = self.norm0(pan)
        output = self.mamba(input)
        output = self.out_proj(output)
        output = rearrange(output, 'b (h w) c -> b c h w', h=h, w=w)
        # output = self.norm2(output)
        return output



