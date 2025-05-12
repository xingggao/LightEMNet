import argparse

from model.mbv2_models import Back_VGG
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



def preprocess_image(image_path, size):
    """ 预处理输入图像 """
    image = Image.open(image_path).convert('RGB')
    orgin_size = image.size
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)  # 添加批次维度
    return image.cuda(),orgin_size  # 移动到GPU


def visualize_and_save(features, save_dir, prefix,orgin_size):
    """ 可视化特征图并保存 """
    feature_names = ['x11', 'x12', 'x13', 'x14', 'x15', 'x1', 'x2', 'x3', 'x4', 'x5', 'edge_map1','x21','x22','x23', 'x24', 'x25','x_conv2','x_conv5','sal_fuse']
    for idx, feature in enumerate(features):
        if feature is not None:  # 确保特征不为空
            feature = feature.squeeze(0)  # 去掉批次维度
            # 计算每个通道的平均值
            average_feature = torch.mean(feature, dim=0).detach().cpu().numpy()
            avg_resized = cv2.resize(average_feature,(orgin_size[0],orgin_size[1]))
            plt.figure(figsize=(10, 10))
            plt.imshow(avg_resized, cmap='viridis')
            plt.axis('off')
            save_path = os.path.join(save_dir, f"{prefix}_{feature_names[idx]}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()


def test_single_image(model, image_path, save_dir, checkpoint_path, size=352):
    """ 测试单张图像并可视化特征图 """
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    image, orgin_size = preprocess_image(image_path, size)

    with torch.no_grad():
        # 前向传播
        x11, x12, x13, x14, x15 = model.backbone(image)
        x_size = image.size()


        x21 = model.upsample2(x11)  # 352*352*64
        x22 = model.upsample2(x12)  # 176*176*128
        x23 = model.upsample2(x13)  # 88*88*256
        x24 = model.upsample2(x14)  # 44*44*512
        x25 = model.upsample2(x15)
        # x11 = model.FA_Block11(x11)
        # x12 = model.FA_Block12(x12)
        # x13 = model.FA_Block13(x13)
        # x14 = model.FA_Block14(x14)
        # x15 = model.FA_Block15(x15)

        # x21 = model.upsample2(x11)
        # x22 = model.upsample2(x12)
        # x23 = model.upsample2(x13)
        # x24 = model.upsample2(x14)
        # x25 = model.upsample2(x15)
        # x6 = model.classification(x5)

        # 进行CRF处理

        # x5 = model.aspp(x25)
        x5 = model.cam5(x25)

        x4 = model.cam4(x24, x5)
        x3 = model.cam3(x23, x4)
        x2 = model.cam2(x22, x3)
        x1 = model.cam1(x21, x2)
        edge_map1 = model.mea(x1, x3, x4)
        edge_map = model.depth(edge_map1)
        # x3_3 = F.interpolate(x3_3,x1.size()[2:],mode='bilinear', align_corners=True)

        # x5 = self.gate(x5)

        # edge_map = self.edge_layer(x1_mea, x3_mea, x4_mea)
        # x33_meaup = self.upsample4(x33_mea)
        # x13 = torch.cat((x11_mea,x33_meaup),dim=1)
        # x134 = self.upsample8(x44_mea)

        # edge_map = self.conv_edge(torch.cat((x13,x134),dim=1))
        # edge_map = self.conv_edge(edge_map)
        # edge_out = torch.sigmoid(edge_map)
        ####

        im_arr = x1.cpu().detach().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)

        canny = torch.from_numpy(canny).cuda().float()
        # cat = torch.cat((edge_out, canny), dim=1)
        acts = torch.cat((edge_map, canny), dim=1)
        # acts = self.fuse_canny_edge(cat)  # 双通道---单通道
        # acts = torch.sigmoid(acts)

        acts1 = F.interpolate(acts, x2.size()[2:], mode='bilinear', align_corners=True)
        acts2 = F.interpolate(acts, x5.size()[2:], mode='bilinear', align_corners=True)
        x4_f = F.interpolate(x4, x5.size()[2:], mode='bilinear', align_corners=True)
        x5_f = model.fusion45(x5, x4_f)
        acts2 = model.depth_conv(acts2)

        x_conv2 = model.fusionmamba2(x2, acts1)
        # x_conv2 = self.FA_Block1(x_conv2)
        x_conv222 = model.depth22(x_conv2)

        x_conv22 = F.interpolate(x_conv2, x5.size()[2:], mode='bilinear', align_corners=True)
        # x_22 = self.depth_conv(x_conv22)
        x_conv5 = model.fusionmamba5(x5_f, acts2)
        # vss5 = torch.cat((x5, acts2), dim=1)

        # x_conv5 = self.FA_Block2(x_conv5)
        x_conv5 = F.interpolate(x_conv5, x2.size()[2:], mode='bilinear', align_corners=True)

        sal_fuse = model.fusionmamba55(x_conv222, x_conv5)
        sal_init = model.FA_Block3(sal_fuse)

        # sal_fuse = self.depth5(sal_fuse)
        #
        # sal_init = self.final_sal_seg(sal_fuse)
        # sal_init = self.final_sal_seg(sal_fuse)
        sal_init = F.interpolate(sal_init, x_size[2:], mode='bilinear')
        # sal_fuse = self.depth5(sal_fuse)
        #
        # sal_init = self.final_sal_seg(sal_fuse)
        # sal_init = self.final_sal_seg(sal_fuse)
        # sal_init = F.interpolate(sal_init, x_size[2:], mode='bilinear')

        # 收集特征图以便可视化
        features = [x11, x12, x13, x14, x15, x1, x2, x3, x4, x5, edge_map1,x21,x22,x23,x24,x25,x_conv2,x_conv5,sal_fuse]

        # 可视化特征图
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        visualize_and_save(features, save_dir, img_name,orgin_size)


if __name__ == '__main__':
    image_path = './S-EOR/train/image/0286.jpg'  # 替换为你的图像路径
    checkpoint_path = './models/mamba_fu/scribble_60.pth'  # 替换为模型权重路径
    save_dir = './temp/signal/fu/0286/'  # 保存可视化结果的目录
    os.makedirs(save_dir, exist_ok=True)

    model = Back_VGG().cuda()  # 替换为你的模型类
    test_single_image(model, image_path, save_dir, checkpoint_path)
    print('Visualization Done!')