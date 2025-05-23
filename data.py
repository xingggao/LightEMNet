import os
from PIL import Image
import torch.utils.data as data
import random
import numpy as np
import torchvision.transforms as transforms
from PIL import ImageEnhance
#several data augumentation strategies
def cv_random_flip(img, label,mask,gray,edge):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        gray = gray.transpose(Image.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, mask, gray, edge


def randomCrop(image, label,mask,gray,edge):
    border=30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), mask.crop(random_region),gray.crop(random_region),edge.crop(random_region)


def randomRotation(image,label,mask,gray,edge):
    mode=Image.BICUBIC
    if random.random()>0.8:
        random_angle = np.random.randint(-15, 15)
        image=image.rotate(random_angle, mode)
        label=label.rotate(random_angle, mode)
        mask = mask.rotate(random_angle, mode)
        gray = gray.rotate(random_angle, mode)
        edge = edge.rotate(random_angle, mode)
    return image,label,mask,gray,edge


def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, mask_root, gray_root, edge_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.masks = [mask_root + f for f in os.listdir(mask_root) if f.endswith('.png')]
        self.grays = [gray_root + f for f in os.listdir(gray_root) if f.endswith('.png')]
        self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.masks = sorted(self.masks)
        self.grays = sorted(self.grays)
        self.edges = sorted(self.edges)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.gray_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.edge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        mask = self.binary_loader(self.masks[index])
        gray = self.binary_loader(self.grays[index])
        edge = self.binary_loader(self.edges[index])
        # image, gt, mask, gray, edge = cv_random_flip(image,gt, mask,gray,edge)
        # image, gt, mask, gray, edge = randomCrop(image, gt, mask, gray, edge)
        # image, gt, mask, gray, edge = randomRotation(image, gt, mask, gray, edge)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        mask = self.mask_transform(mask)
        gray = self.gray_transform(gray)
        edge = self.edge_transform(edge)
        return image, gt, mask, gray, edge

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        masks = []
        grays = []
        edges = []
        for img_path, gt_path, mask_path, gray_path, edge_path in zip(self.images, self.gts, self.masks, self.grays,  self.edges):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            mask = Image.open(mask_path)
            gray = Image.open(gray_path)
            edge = Image.open(edge_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                masks.append(mask_path)
                grays.append(gray_path)
                edges.append(edge_path)
        self.images = images
        self.gts = gts
        self.masks = masks
        self.grays = grays
        self.edges = edges

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, mask_root, gray_root, edge_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True):

    dataset = SalObjDataset(image_root, gt_root, mask_root, gray_root, edge_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        HH = image.size[0]
        WW = image.size[1]
        image = self.transform(image).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, HH, WW, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

