# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/12/9 下午1:54

from unet_model.unet import UNET
from unet_model.Attention_UNet import *
from dataset_DRIVE import DRIVE_Dataset
from Dice import DiceLoss
import torch.optim as optim
import torch.nn as nn
import torch
import CosineAnnealingWithWarmup as LR

def train_net(net,device,data_path,epochs=300,batch_size=1,lr=4e-5):
    drive_dataset = DRIVE_Dataset(data_path)
    train_loader = torch.utils.data.DataLoader(drive_dataset,
                                               batch_size,
                                               shuffle = True)
    #使用RMSprop优化
    # optimizer = optim.RMSprop(net.parameters(),lr,weight_decay=1e-8,momentum=0.9)
    optimizer = optim.AdamW(net.parameters(),lr,weight_decay=1e-8)
    criterion = DiceLoss()
    # criterion = nn.
    best_loss = float("inf")
    coslr = LR.LR_Scheduler(optimizer=optimizer, warmup_epochs=5, warmup_lr=lr*10, num_epochs=epochs, base_lr=lr, final_lr=lr/100, iter_per_epoch=1)
    for epoch in range(epochs):
        net.train()
        coslr.step()
        
        for images, labels in train_loader:
            optimizer.zero_grad()

            images = images.to(device,dtype = torch.float32)
            # print(images.shape)
            labels = labels.to(device,dtype=torch.float32)
            pred = net(images)
            # import pdb
            # pdb.set_trace()
            loss = criterion(pred,labels)
            print('epoch:%d  train loss:%f coslr:%f' % (epoch+1,loss.item(),coslr.get_lr()))
            if loss <best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    device = torch.device('cuda:0')
    net = UNET(n_channels=1,n_classes=1)
    # net = R2U_Net(img_ch=1,output_ch=1)
    
    net.to(device)
    data_path = "DRIVE/training/"
    train_net(net = net, device = device, data_path = data_path, batch_size = 2)
