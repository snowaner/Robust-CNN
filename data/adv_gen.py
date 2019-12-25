#! /usr/bin/env python
#################################################################################
#     File Name           :     adv_gen.py
#     Created By          :     common
#     Creation Date       :     [2019-12-25 06:53]
#     Last Modified       :     [2019-12-25 09:31]
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

# import by relative path
sys.path.append('..')
from models import ResNet18

# inner parameters
data_path = '/userhome/wancq/cifar10/data/'
pretrained_model = '/userhome/wancq/Robust-CNN/model/base.pth'
epsilon = 0.1
use_cuda = True
save_adv_examples = True
save_path = '/userhome/wancq/Robust-CNN/data/raw/adv/'

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

# training related
criterion = nn.CrossEntropyLoss()

# fgsm attack
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    #perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def adv_test(dataloader, epsilon, model, criterion):
    correct = 0
    count = 0
    adv_examples = []
    for data, target in dataloader:
        count += 1
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        init_flag = 1 - torch.sign(init_pred[:,0] - target).abs()
        loss = criterion(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        data_grad = data_grad * init_flag.view(-1,1,1,1).expand(data_grad.size()).float()
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        final_flag = (1 - torch.sign(final_pred[:,0] - target).abs())
        correct += final_flag.sum()
        for i in range(perturbed_data.size(0)):
            if init_flag[i].item() == 1 and final_flag[i].item() == 0:
                adv_examples.append(perturbed_data[i].detach().cpu().numpy())
        print('{} batches has been processed'.format(count))
    final_acc = float(correct)/(1000*len(dataloader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, 1000*len(dataloader), final_acc))
    np.save('eps-01.npy', np.asarray(adv_examples))
    print('{} adversary examples have been saved'.format(len(adv_examples)))

adv_test(trainloader, epsilon, model, criterion)
