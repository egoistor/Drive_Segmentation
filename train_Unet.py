# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/12/9 下午1:54

from unet_model.unet import UNET
from dataset_DRIVE import DRIVE_Dataset
from Dice import DiceLoss
import torch.optim as optim
import torch.nn as nn
import torch


def train_net(net,device,data_path,epochs=300,batch_size=2,lr=1e-5):
    drive_dataset = DRIVE_Dataset(data_path)
    train_loader = torch.utils.data.DataLoader(drive_dataset,
                                               batch_size,
                                               shuffle = True)
    #使用RMSprop优化
    optimizer = optim.RMSprop(net.parameters(),lr,weight_decay=1e-8,momentum=0.9)
    criterion = DiceLoss()
    # criterion = nn.
    best_loss = float("inf")

    for epoch in range(epochs):
        net.train()
        for images, labels in train_loader:
            optimizer.zero_grad()

            images = images.to(device,dtype = torch.float32)
            print(images.shape)
            labels = labels.to(device,dtype=torch.float32)
            pred = net(images)

            loss = criterion(pred,labels)
            print('epoch:%d  train loss:%f' % (epoch+1,loss.item()))
            if loss <best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    device = torch.device('cuda:0')
    net = UNET(n_channels=1,n_classes=1)
    net.to(device)
    data_path = "DRIVE/training/"
    train_net(net,device,data_path)
