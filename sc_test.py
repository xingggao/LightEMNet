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
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)  # 添加批次维度
    return image.cuda()  # 移动到GPU


def visualize_and_save(features, save_dir, prefix):
    """ 可视化特征图并保存 """
    feature_names = ['x1_crf', 'x2_crf', 'x3_crf', 'x4_crf', 'x5_crf', 'edge_map', 'x6', 'x11', 'x12', 'x13', 'x14', 'x15']
    for idx, feature in enumerate(features):
        if feature is not None:  # 确保特征不为空
            feature = feature.squeeze(0)  # 去掉批次维度
            # 计算每个通道的平均值
            average_feature = torch.mean(feature, dim=0).detach().cpu().numpy()
            plt.figure(figsize=(10, 10))
            plt.imshow(average_feature, cmap='viridis')
            plt.axis('off')
            save_path = os.path.join(save_dir, f"{prefix}_{feature_names[idx]}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()


def test_single_image(model, image_path, save_dir, checkpoint_path, size=352):
    """ 测试单张图像并可视化特征图 """
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    image = preprocess_image(image_path, size)

    with torch.no_grad():
        # 前向传播
        x11, x12, x13, x14, x15 = model.backbone(image)
        x1 = model.upsample2(model.conv1(x11))
        x2 = model.upsample2(model.conv2(x12))
        x3 = model.upsample2(model.conv3(x13))
        x4 = model.upsample2(model.conv4(x14))
        x5 = model.upsample2(model.conv5(x15))
        x6 = model.classification(x5)

        # 进行CRF处理

        x5_crf = model.crf5(x5, x6)
        x4_crf = model.crf4(x4, x6)
        x3_crf = model.crf3(x3, x6)
        x2_crf = model.crf2(x2, x6)
        x1_crf = model.crf1(x1, x6)
        edge_map = model.mea(x1_crf, x3_crf, x4_crf)

        # 收集特征图以便可视化
        features = [x1_crf, x2_crf, x3_crf, x4_crf, x5_crf, edge_map, x6, x11, x12, x13, x14, x15]

        # 可视化特征图
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        visualize_and_save(features, save_dir, img_name)


if __name__ == '__main__':
    image_path = './S-EOR/test/image/0014.jpg'  # 替换为你的图像路径
    checkpoint_path = './models/scribble_60.pth'  # 替换为模型权重路径
    save_dir = './temp/signal_image/'  # 保存可视化结果的目录
    os.makedirs(save_dir, exist_ok=True)

    model = Back_VGG().cuda()  # 替换为你的模型类
    test_single_image(model, image_path, save_dir, checkpoint_path)
    print('Visualization Done!')