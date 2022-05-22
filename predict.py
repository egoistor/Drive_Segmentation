# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/12/9 下午2:07
import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import cv2
from unet_model.unet import UNET
from unet_model.Attention_UNet import *
import re
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
if __name__ == "__main__":
    device = torch.device('cuda:0')
    # net = UNET(n_channels=1,n_classes=1)
    net = R2U_Net(img_ch=1,output_ch=1)
    net = net.to(device)
    net.load_state_dict(torch.load('best_model.pth'))
    net.eval()
    testpaths = glob.glob(r'DRIVE/test/images/*.tif')
    count = 0
    for test_path in testpaths:
        test_path = re.sub(r'\\', '/', test_path)
        print(test_path)
        save_res_path = 'result/' + str(count) + '.png'
        save_res_path = re.sub(r'\\', '/', save_res_path)
        print(save_res_path)
        img = cv2.imread(test_path,-1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(1,1,img.shape[0],img.shape[1])
        img = torch.from_numpy(img)
        img = img.to(device,dtype = torch.float32)

        pred = net(img)
        pred = np.array(pred.data.cpu()[0])[0]

        #从二值还原为灰度图
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # plt.imshow(pred)
        # plt.show()
        cv2.imwrite(save_res_path,pred)
        count = count + 1
