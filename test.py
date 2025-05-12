import cv2
import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
# from scipy import misc

from model.mbv2_models import Back_VGG
from data import test_dataset
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
opt = parser.parse_args()

dataset_path = './S-EOR/test/image/'
# dataset_path = './S-EOR/Image-test/'
model = Back_VGG(channel=32)
model.load_state_dict(torch.load('./models/mamba_fu/scribble_60.pth'))

model.cuda()
model.eval()
from thop import profile
def noop(module, input, output):
    return output
from thop.vision.basic_hooks import zero_ops
input1 = torch.randn(1,3,352,352).to("cuda")
flops, params = profile(model, (input1,), custom_ops={torch.nn.ReLU: noop})
params = params / 1e6
print(f'params: {params:.2f} M')
print('flops: ', flops, 'params: ', params)

# test_datasets = ['ECSSD', 'DUT', 'DUTS_Test', 'THUR', 'HKU-IS']
test_datasets = ['EORSSD']
for dataset in test_datasets:
    save_path = './results/mamba_fu2/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path
    test_loader = test_dataset(image_root, opt.testsize)
    img_list = []
    time_list = []
    for i in range(test_loader.size):
        print(i)
        start = time.time()
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        res0, res1, res, x5,x4,x3,x2,x1,x11, x12, x13, x14,x15,x1_1,x1_2,x1_3,x1_4,x1_5,x_conv2,x_conv5,sal_fuse= model(image)
        end = time.time()
        time_list.append((end - start))
        res = F.upsample(res, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        cv2.imwrite(save_path+name, res * 255)
    time_cost = float(sum(time_list) / len(time_list))
    print('FPS: {:.5f}'.format(time_cost))