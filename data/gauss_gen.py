#! /usr/bin/env python
#################################################################################
#     File Name           :     adv_gen.py
#     Created By          :     common
#     Creation Date       :     [2019-12-25 06:53]
#     Last Modified       :     [2019-12-25 12:14]
#     Description         :     adversary examples generation by FGSM
#################################################################################

from __future__ import print_function
import sys
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import numpy as np
import random

# import by relative path
sys.path.append('..')
from models import ResNet18

# inner parameters
data_path = '/userhome/wancq/cifar10/data/'
pretrained_model = '/userhome/wancq/Robust-CNN/model/base.pth'
epsilon = 0.5
use_cuda = True
save_adv_examples = True
save_path = '/userhome/wancq/Robust-CNN/data/raw/gauss/'

# data preparing
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=False, num_workers=0)

testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0)

# cuda settings
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# model definition
model = ResNet18().to(device)
model = torch.nn.DataParallel(model)
ckp = torch.load(pretrained_model)
model.load_state_dict(ckp['net'])
model.eval()

def gauss_test(dataloader, epsilon, model):
    correct = 0
    count = 0
    gauss_examples = []
    for data, target in dataloader:
        count += 1
        data, target = data.to(device), target.to(device)
        noise = np.zeros(data.numel())
        for i in range(data.numel()):
            noise[i] = random.gauss(0, 1)
        noise = torch.autograd.Variable(torch.from_numpy(noise).reshape(data.size())).float().cuda()
        noise_data = data + epsilon * noise
        output = model(noise_data)
        final_pred = output.max(1, keepdim=True)[1]
        final_flag = (1 - torch.sign(final_pred[:,0] - target).abs())
        correct += final_flag.sum()
        for i in range(data.size(0)):
            if final_flag[i].item() == 0:
                gauss_examples.append(noise_data[i].cpu().numpy())
        print('{} batches has been processed'.format(count))
    final_acc = float(correct)/(1000*len(dataloader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, 1000*len(dataloader), final_acc))
    np.save('eps-05.npy', np.asarray(gauss_examples))
    print('{} gauss noise examples have been saved'.format(len(gauss_examples)))

gauss_test(trainloader, epsilon, model)
