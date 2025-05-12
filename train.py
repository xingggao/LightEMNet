import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torchsummary import summary
import numpy as np
import pdb, os, argparse
from datetime import datetime
import logging
from model.mbv2_models import Back_VGG
from data import get_loader
from utils import clip_gradient,adjust_lr
import os
import smoothness
from model.LocalSaliencyCohence import LocalSaliencyCoherence
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=60, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=6, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=20, help='every n epochs decay learning rate')
parser.add_argument('--sm_loss_weight', type=float, default=0.3, help='weight for smoothness loss')
parser.add_argument('--edge_loss_weight', type=float, default=1.0, help='weight for edge loss')
parser.add_argument('--log_save_path', type=str, default='./results/logs/', help='path for log')
opt = parser.parse_args()

print('Learning Rate: {}'.format(opt.lr))
# build models
model = Back_VGG(channel=32)
if not os.path.exists(opt.log_save_path):
    os.makedirs(opt.log_save_path)
logging.basicConfig(filename=opt.log_save_path + 'mobilenet.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("mobilenet_pairs")
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
# # summary(model,(3,352,352))

# from thop import profile
# from thop import clever_format
# input1 = torch.randn(1,3,352,352).to("cuda")
# flops, params = profile(model, (input1,))
# params = params / 1e6
# print(f'params: {params:.2f} M')
# print('flops: ', flops, 'params: ', params)


logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};log_save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.log_save_path, opt.decay_epoch))


image_root = './S-EOR/train/image/'
gt_root = './S-EOR/train/gt/'
mask_root = './S-EOR/train/mask/'
edge_root = './S-EOR/train/edge/'
grayimg_root = './S-EOR/train/gray/'


# image_root1 = './S-EOR/test/image/'
# gt_root1 = './S-EOR/test/gt/'
# mask_root1 = './S-EOR/test/mask/'
# scribble = './S-EOR/test/scribble/'

train_loader = get_loader(image_root, gt_root, mask_root, grayimg_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCELoss()
CEL = torch.nn.CrossEntropyLoss()
loss_lsc = LocalSaliencyCoherence()
smooth_loss = smoothness.smoothness_loss(size_average=True)
loss_lsc_kernels_desc_defaults = [{"weight":1, "xy":6, "rgb":0.1}]

def visualize_prediction1(pred):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_sal1.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_prediction2(pred):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_sal2.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_edge(pred):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_edge.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)


def train(train_loader, model, optimizer, epoch):
    model.train()
    loss_all = 0
    epoch_step = 0
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts, masks, grays, edges = pack
        images = Variable(images)
        gts = Variable(gts)
        masks = Variable(masks)
        grays = Variable(grays)
        edges = Variable(edges)
        images = images.cuda()
        gts = gts.cuda()
        masks = masks.cuda()
        grays = grays.cuda()
        edges = edges.cuda()

        img_size = images.size(2) * images.size(3) * images.size(0)
        ratio = img_size / torch.sum(masks)
        sample = {'rgb':images}
        gt = masks.squeeze(1).long()
        bg_label = gt.clone()
        fg_label = gt.clone()
        bg_label[gt !=0] =255
        fg_label[gt == 0] = 255

        # sal1, edge_map, sal2 = model(images)    # sal_init, edge_map, sal_ref = model(images)
        sal1, edge_map, sal2,x5,x4,x3,x2,x1,x11, x12, x13, x14,x15,x1_1,x1_2,x1_3,x1_4,x1_5,x_conv2,x_conv5,sal_fuse= model(images)
        sal1_prob = torch.sigmoid(sal1)
        sal1_prob = sal1_prob * masks
        sal2_prob = torch.sigmoid(sal2)
        sal2_prob = sal2_prob * masks

        smoothLoss_cur1 = opt.sm_loss_weight * smooth_loss(torch.sigmoid(sal1), grays)
        CEL1 = CEL(sal1_prob,gts*masks) + CEL(sal1_prob,gts*masks)
        # loss1_lsc = loss_lsc(sal_loss1, loss_lsc_kernels_desc_defaults, 5, sample, images.shape[2], images.shape[3])['loss']
        sal_loss1 = ratio * CE(sal1_prob, gts*masks) + smoothLoss_cur1 + CEL1

        smoothLoss_cur2 = opt.sm_loss_weight * smooth_loss(torch.sigmoid(sal2), grays)
        CEL2 = CEL(sal1_prob, gts*masks) + CEL(sal1_prob, gts*masks)
        # loss2_lsc = loss_lsc(sal_loss2, loss_lsc_kernels_desc_defaults, 5, sample, images.shape[2], images.shape[3])['loss']
        sal_loss2 = ratio * CE(sal2_prob, gts * masks) + smoothLoss_cur2 + CEL2
        edge_loss = opt.edge_loss_weight*CE(torch.sigmoid(edge_map),edges)

        bce = sal_loss1 + edge_loss + sal_loss2


        visualize_prediction1(torch.sigmoid(sal1))
        visualize_edge(torch.sigmoid(edge_map))
        visualize_prediction2(torch.sigmoid(sal2))


        loss = bce
        loss.backward()

        epoch_step += 1
        loss_all += loss.data
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 10 == 0 or i == total_step:
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], sal1_loss: {:0.4f}, edge_loss: {:0.4f}, sal2_loss: {:0.4f},  loss: {:0.4f}'.
                format(datetime.now(), epoch, opt.epoch, i, total_step, sal_loss1.data, edge_loss.data, sal_loss2.data,
                       loss.data))
            logging.info(
                '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}, sal_loss1:{:4f}, sal_loss2:{:4f}, '
                'edge_loss:{:4f}, loss:{:4f}, mem_use:{:.0f}MB'.
                format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'],
                       sal_loss1.data, sal_loss2.data, edge_loss.data,  loss.data, memory_used))
        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))


    save_path = 'models/mamba_fu/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % 10 == 0:
        torch.save(model.state_dict(), save_path + 'scribble' + '_%d'  % epoch  + '.pth')


print("Scribble it!")
for epoch in range(1, opt.epoch+1):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)

