'''
CUDA_VISIBLE_DEVICES=4,5 python -m test_dp02.py
'''
import argparse
import time
import torch
import torchvision
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import SequentialSampler
import numpy as np

batch_size = 128
epochs = 1
lr = 0.001

from torchvision.models.resnet import ResNet, BasicBlock


class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        return torch.softmax(super(MnistResNet, self).forward(x), dim=-1)


# net = resnet18()
net = MnistResNet()
net.cuda()
net = DP(net)


class ToNumpy(object):
    def __call__(self, sample):
        return np.array(sample)


data_root = 'dataset'
trainset = MNIST(root=data_root,
                 download=True,
                 train=True,
                 transform=torchvision.transforms.Compose(
                     [ToNumpy(), torchvision.transforms.ToTensor()])
                 )

valset = MNIST(root=data_root,
               download=True,
               train=False,
               transform=torchvision.transforms.Compose(
                   [ToNumpy(), torchvision.transforms.ToTensor()])
               )

sampler = SequentialSampler(trainset)
train_loader = DataLoader(trainset,
                          batch_size=batch_size,
                          shuffle=False,
                          pin_memory=True,
                          sampler=sampler)

val_loader = DataLoader(valset,
                        batch_size=batch_size,
                        shuffle=False,
                        pin_memory=True)

criterion = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr=lr)

net.train()
for e in range(epochs):
    # DistributedSampler deterministically shuffle data
    # by seting random seed be current number epoch
    # so if do not call set_epoch when start of one epoch
    # the order of shuffled data will be always same
    for idx, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.cuda()
        labels = labels.cuda()
        output = net(imgs)
        loss = criterion(output, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if idx % 10 == 0:
            print('Epoch: {} step: {} loss: {}'.format(e, idx, loss.item()))
net.eval()
with torch.no_grad():
    cnt = 0
    total = len(val_loader.dataset)
    for imgs, labels in val_loader:
        imgs, labels = imgs.cuda(), labels.cuda()
        output = net(imgs)
        predict = torch.argmax(output, dim=1)
        cnt += (predict == labels).sum().item()

# count:9081, total:10000, eval accuracy: 0.9081
# print only once
print('count:{}, total:{}, eval accuracy: {}'.format(cnt, total, cnt / total))
