import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
from model.down import Block
from model.MobileNetV2 import mobilenet_v2
from math import log
from model.dirc import DirectionalConvUnit
from model.mamba import CAMMambaBlock, FusionMamba

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
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

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

class MEA(nn.Module):
    def __init__(self, channel1,channel2,channel3):
        super(MEA, self).__init__()

        self.conv1_1 = BasicConv2d(channel1, channel1, 1)
        self.conv1_3 = BasicConv2d(channel1, channel1, 3, padding=1)
        self.conv2_1 = BasicConv2d(channel2, channel2, 1)
        self.conv2_3 = BasicConv2d(channel2, channel2, 3, padding=1)
        self.conv3_1 = BasicConv2d(channel3, channel3, 1)
        self.conv3_3 = BasicConv2d(channel3, channel3, 3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.agvpool = nn.AvgPool2d(kernel_size=2,stride=2)
        self.upsample2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.conv_up = BasicConv2d(channel1*2, channel2, 3, padding=1)   #128->256
        self.conv_down = BasicConv2d(channel3, channel2, 3, padding=1)    #512->
        self.ca = ChannelAttention(channel3)
        self.rcab = RCAB(channel2)
        self.edge = nn.Sequential(
            BasicConv2d(channel2, channel2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel2, 1, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1,x2,x3):
        x1 = self.conv1_3(self.conv1_1(x1))
        x2 = self.conv2_3(self.conv2_1(x2))
        x3 = self.conv3_3(self.conv3_1(x3))
        x1_1 = self.maxpool(x1)
        x1_2 = self.conv_up(torch.cat((x1_1,self.agvpool(x1)),1))

        x1_3 = F.interpolate(x2,x1_2.size()[2:],mode='bilinear', align_corners=True)
        x2_1 = x1_3 * x1_2
        x3 =  F.interpolate(self.conv_down(self.ca(x3)),x2_1.size()[2:],mode='bilinear', align_corners=True)
        x3_3 = F.interpolate(x2_1 * x3,x1.size()[2:],mode='bilinear', align_corners=True)
        x_edge = self.edge(self.rcab(x3_3))
        return self.sigmoid(x_edge)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        # 深度可分离卷积包括深度卷积和逐点卷积两个步骤
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                                   stride=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0,
                                   stride=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class CAM(nn.Module):
    def __init__(self, hchannel, channel):
        super(CAM, self).__init__()
        self.conv1_1 = Conv1x1(hchannel + channel, channel)
        self.conv3_1 = ConvBNR(channel // 4, channel // 4, 3)
        self.dconv5_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=2)
        self.dconv7_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=3)
        # self.dconv9_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=4)
        self.dconv9_1 = CAMMambaBlock(channel // 4)
        self.conv1_2 = Conv1x1(channel, channel)
        self.conv3_3 = ConvBNR(channel, hchannel, 3)

    def forward(self, lf, hf):
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((lf, hf), dim=1)
        x = self.conv1_1(x)
        xc = torch.chunk(x, 4, dim=1)
        x0 = self.conv3_1(xc[0] + xc[1])
        x1 = self.dconv5_1(xc[1] + x0 + xc[2])
        x2 = self.dconv7_1(xc[2] + x1 + xc[3])
        x3 = self.dconv9_1(xc[3] + x2)
        xx = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.conv3_3(x + xx)

        return x

class CAM2(nn.Module):
    def __init__(self,channel):
        super(CAM2, self).__init__()
        self.depth4 = DepthwiseSeparableConv(channel+192, channel)
        self.conv3_1 = ConvBNR(channel // 4, channel // 4, 3)
        self.dconv5_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=2)
        self.dconv7_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=3)
        # self.dconv9_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=4)
        self.dconv9_1 = CAMMambaBlock(channel // 4)
        self.conv1_2 = Conv1x1(channel, channel)
        self.conv3_3 = ConvBNR(channel, channel, 3)

    def forward(self, x):
        xc = torch.chunk(x, 4, dim=1)
        x0 = self.conv3_1(xc[0] + xc[1])
        x1 = self.dconv5_1(xc[1] + x0 + xc[2])
        x2 = self.dconv7_1(xc[2] + x1 + xc[3])
        x3 = self.dconv9_1(xc[3] + x2)
        xx = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.conv3_3(x + xx)

        return x


class Back_MBV2(nn.Module):
    def __init__(self, pretrained = True,channel=32):
        super(Back_VGG, self).__init__()
        self.backbone = mobilenet_v2(pretrained)
        self.mea = MEA(channel1=16,channel2=32,channel3=96)

        self.conv_edge = nn.Conv2d(3,1,1, bias=False)
        self.cam4 = CAM(96, 320)
        self.cam3 = CAM(32, 96)
        self.cam5 = CAM2(320)
        self.cam2 = DirectionalConvUnit(24, 32)
        self.cam1 = DirectionalConvUnit(16, 24)
        self.depth_conv = DepthwiseSeparableConv(24, 320)
        self.depth = DepthwiseSeparableConv(1, 23)
        self.depth22 = DepthwiseSeparableConv(24, 320)

        self.fusionmamba2 = FusionMamba(24)
        self.fusionmamba5 = FusionMamba(320)
        self.fusionmamba55 = FusionMamba(320)
        self.FA_Block = Block(in_dim=320, out_dim=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.rcab_sal_edge = RCAB(24)
        self.fused_edge_sal = nn.Conv2d(24, 1, kernel_size=3, padding=1, bias=False)


    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x):
        x_size = x.size()
        x1_1, x1_2, x1_3, x1_4,x1_5 = self.backbone(x)

        x11 = self.upsample2(x1_1)  # 352*352*64
        x12 = self.upsample2(x1_2)  # 176*176*128
        x13 = self.upsample2(x1_3)  # 88*88*256
        x14 = self.upsample2(x1_4)  # 44*44*512
        x15 = self.upsample2(x1_5)  # 22*22*512

        x5 = self.cam5(x15)
        x4 = self.cam4(x14, x5)
        x3 = self.cam3(x13, x4)
        x2 = self.cam2(x12, x3)
        x1 = self.cam1(x11, x2)


        edge_map1 = self.mea(x1,x3,x4)
        edge_map = self.depth(edge_map1)
        im_arr = x1.cpu().detach().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)

        canny = torch.from_numpy(canny).cuda().float()

        acts = torch.cat((edge_map, canny), dim=1)

        acts1 = F.interpolate(acts, x2.size()[2:], mode='bilinear', align_corners=True)
        acts2 = F.interpolate(acts, x5.size()[2:], mode='bilinear', align_corners=True)
        x4_f = F.interpolate(x4, x5.size()[2:], mode='bilinear', align_corners=True)
        acts2 = self.depth_conv(acts2)

        x_conv2 = self.fusionmamba2(x2, acts1)
        x_conv222 = self.depth22(x_conv2)

        x_conv22 = F.interpolate(x_conv2, x5.size()[2:], mode='bilinear', align_corners=True)
        x_conv5 = self.fusionmamba5(x5, acts2)

        x_conv5 = F.interpolate(x_conv5, x2.size()[2:], mode='bilinear', align_corners=True)

        sal_fuse = self.fusionmamba55(x_conv222,x_conv5)
        sal_init = self.FA_Block(sal_fuse)

        sal_init = F.interpolate(sal_init, x_size[2:], mode='bilinear')

        sal_edge_feature = torch.cat((sal_init, edge_map), 1)
        sal_edge_feature = self.rcab_sal_edge(sal_edge_feature)
        sal_ref = self.fused_edge_sal(sal_edge_feature)
        return sal_init, edge_map1, sal_ref, x5, x4, x3, x2, x1, x11, x12, x13, x14, x15,x1_1,x1_2,x1_3,x1_4,x1_5,x_conv2,x_conv5,sal_fuse