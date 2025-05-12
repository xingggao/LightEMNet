import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mamba import CAMMambaBlock,DirMambaBlock
from model.down import Block
class Conv1x1(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,1)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class DirectionalConvUnit(nn.Module):
    def __init__(self, channel, hchannel):
        super(DirectionalConvUnit, self).__init__()
        self.conv1_1 = Conv1x1(hchannel + channel, channel)

        self.h_conv = nn.Conv2d(channel, channel // 4, (1, 5), padding=(0, 2))
        self.w_conv = nn.Conv2d(channel, channel // 4, (5, 1), padding=(2, 0))
        # leading diagonal
        self.dia19_conv = nn.Conv2d(channel, channel // 4, (5, 1), padding=(2, 0))
        # reverse diagonal
        self.dia37_conv = nn.Conv2d(channel, channel // 4, (1, 5), padding=(0, 2))

    def forward(self, x, x2):
        if x.size()[2:] != x2.size()[2:]:
            x2 = F.interpolate(x2, size=x.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, x2), dim=1)
        x = self.conv1_1(x)
        x1 = self.h_conv(x)
        x2 = self.w_conv(x)
        x3 = self.inv_h_transform(self.dia19_conv(self.h_transform(x)))
        x4 = self.inv_v_transform(self.dia37_conv(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)

        return x

    # Code from "CoANet- Connectivity Attention Network for Road Extraction From Satellite Imagery", and we modified the code
    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-2]]
        x = x.reshape(shape[0], shape[1], shape[2], shape[2]+shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[2], shape[3]+1)
        x = x[..., 0: shape[3]-shape[2]+1]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-2]]
        x = x.reshape(shape[0], shape[1], shape[2], shape[2]+shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[2], shape[3]+1)
        x = x[..., 0: shape[3]-shape[2]+1]
        return x.permute(0, 1, 3, 2)

def antidiagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (torch.arange(W, device=tensor.device) - shift) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index)

def diagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (shift + torch.arange(W, device=tensor.device)) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index)

def CrossScan(x: torch.Tensor):
    B, C, H, W = x.shape
    xc = torch.chunk(x, 4, dim=1)

    xs = x.new_empty((B, 4, C//4, H, W))
    # 添加横向和竖向的扫描
    xs[:, 0] = xc[0]
    xs[:, 1] = xc[1].transpose(dim0=2, dim1=3)
    # xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])

    # 提供斜向和反斜向的扫描
    xs[:, 2] = diagonal_gather(xc[2])
    xs[:, 3] = antidiagonal_gather(xc[3])
    # xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

    return xs


def diagonal_scatter(tensor_flat, original_shape):
    # 把斜向元素拼接起来的一维向量还原为最初的矩阵形式
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (shift + torch.arange(W, device=tensor_flat.device)) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 创建一个空的张量来存储反向散布的结果
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # 将平铺的张量重新变形为[B, C, H, W]，考虑到需要使用transpose将H和W调换
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # 使用scatter_根据expanded_index将元素放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor

def antidiagonal_scatter(tensor_flat, original_shape):
    # 把反斜向元素拼接起来的一维向量还原为最初的矩阵形式
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (torch.arange(W, device=tensor_flat.device) - shift) % W  # 利用广播创建索引矩阵[H, W]
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 初始化一个与原始张量形状相同、元素全为0的张量
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # 将平铺的张量重新变形为[B, C, W, H]，因为操作是沿最后一个维度收集的，需要调整形状并交换维度
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # 使用scatter_将元素根据索引放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor






class Directionalmamba(nn.Module):
    def __init__(self, channel, hchannel):
        super(Directionalmamba, self).__init__()
        self.conv1_1 = Conv1x1(hchannel + channel, channel)
        self.mamba = CAMMambaBlock(channel//4)
        self.conv1_2 = Conv1x1(channel*4, channel)
        self.h_conv = nn.Conv2d(channel, channel//4, (1, 5), padding=(0, 2))
        self.w_conv = nn.Conv2d(channel, channel//4, (5, 1), padding=(2, 0))
        # leading diagonal
        self.dia19_conv = nn.Conv2d(channel, channel // 4, (5, 1), padding=(2, 0))
        # reverse diagonal
        self.dia37_conv = nn.Conv2d(channel, channel // 4, (1, 5), padding=(0, 2))


    def forward(self, x, x2):
        if x.size()[2:] != x2.size()[2:]:
            x2 = F.interpolate(x2, size=x.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, x2), dim=1)
        x = self.conv1_1(x)
        xs = CrossScan(x)
        B, C, H, W = xs[:, 0, :, :, :].shape
        # x1 = self.inv_h_transform(self.dia19_conv(self.h_transform(x)))
        # x2 = self.inv_v_transform(self.dia37_conv(self.v_transform(x)))
        x1 = self.h_conv(x)
        x2 = self.w_conv(x)
        x3 = self.inv_h_transform(self.dia19_conv(self.h_transform(x)))
        x4 = self.inv_v_transform(self.dia37_conv(self.v_transform(x)))
        x1 = self.mamba(xs[:, 0, :, :, :]) + x1
        x2 = self.mamba(xs[:, 1, :, :, :]).transpose(dim0=2, dim1=3) + x2
        x3 = diagonal_scatter(self.mamba(xs[:, 2, :, :, :]), (B,C,H,W)) + x3
        x4 = antidiagonal_scatter(self.mamba(xs[:, 3, :, :, :]),(B,C,H,W)) + x4

        x = torch.cat((x1, x2, x3, x4), 1)


        return x

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-2]]
        x = x.reshape(shape[0], shape[1], shape[2], shape[2]+shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[2], shape[3]+1)
        x = x[..., 0: shape[3]-shape[2]+1]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-2]]
        x = x.reshape(shape[0], shape[1], shape[2], shape[2]+shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[2], shape[3]+1)
        x = x[..., 0: shape[3]-shape[2]+1]
        return x.permute(0, 1, 3, 2)