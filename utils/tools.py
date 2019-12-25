#! /usr/bin/env python
#################################################################################
#     File Name           :     tools.py
#     Created By          :     longyj
#     Creation Date       :     [2018-12-11 22:39]
#     Last Modified       :     [2019-12-25 13:01]
#     Description         :      
#################################################################################

import os
import sys
import errno
import os.path as osp
import random
import math
import time
import pdb
import torchvision
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms import *
from torch.nn import init
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

def load_as_numpy(root, data_list):
    datasets = {}
    for dl in data_list:
        img = Image.open(osp.join(root,dl[0])).convert('RGB')
        datasets[dl[0]] = img
    return datasets

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def readtxt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    ret = []
    for line in lines:
        fname, pid, cam = line.split()
        ret.append((fname, int(pid), int(cam)))
    return ret

def adjust_lr(base_lr, epoch, steps, optimizer):
    lr = base_lr
    for step in steps:
        if epoch > step:
            lr = base_lr * 0.1
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr_mult', 1)

def extract(model, imgs):
    model.eval()
    return model(imgs)

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1. / batch_size))
    return ret

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        angle = None
        fname, pid, cam = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, cam

class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)

class RandomSizedRectCrop(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.64, 1.0) * area
            aspect_ratio = random.uniform(2, 3)

            h = int(round(math.sqrt(target_area*aspect_ratio)))
            w = int(round(math.sqrt(target_area/aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1+w, y1+h))
                assert(img.size == (w, h))

                return img.resize((self.width, self.height), self.interpolation)

        scale = RectScale(self.height, self.width,
                          interpolation=self.interpolation)
        return scale(img)

class RandomErasing(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, rl=0.3,
                 mean=[0.4914,0.4822,0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.rl = rl

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attemp in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.rl, 1/self.rl)

            h = int(round(math.sqrt(target_area*aspect_ratio)))
            w = int(round(math.sqrt(target_area/aspect_ratio)))

            if w <= img.size()[1] and h <= img.size()[2]:
                x1 = random.randint(0, img.size()[1] - w)
                y1 = random.randint(0, img.size()[2] - h)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img
        return img

class RandomRotation(object):
    def __init__(self, angles=None):
        if angles is None:
            self.angles = [0]
            self.angle = 0
        else:
            self.angles = angles
            self.angle = None

    def __call__(self, img):
        index = random.randint(0, len(self.angles)-1)
        img = img.rotate(self.angles[index])
        self.angle = self.angles[index]
        return img

