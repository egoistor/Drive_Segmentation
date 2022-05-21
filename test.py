import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import imageio
import glob
from torch.utils.data import Dataset
import random


def deal_data():
    img_path = r'DRIVE/training/images/21_training.tif'
    manual_path = r'DRIVE/training/1st_manual/21_manual1.gif'
    roi_path = r'DRIVE/training/mask/21_training_mask.gif'
    img = cv2.imread(img_path,-1)

    manual = Image.open(manual_path).convert('L')

    # 在设置mask中的类别时，0是背景，1是前景，而此时的mask中的前景像素值是255，所以÷255，令其为1
    # 此时mask中的类别就是从0开始的（连续）
    manual = np.array(manual) #/ 255

    # roi_mask的图像，并转化成灰度图
    roi_mask = Image.open(roi_path).convert('L')
    # 将不感兴趣区域的no_roi区域的像素值设置成255（不参与计算LOSS）
    roi_mask = 255 - np.array(roi_mask)

    # 使用np.clip()方法，为叠加了manual(GT)与roi_mask后的像素设置像素的上下限
    label = np.clip(manual + roi_mask, a_min=0, a_max=255)/255
    print(label)
    # 此时的感兴趣区域中的前景像素值=1，后景=0，不感兴趣区域像素值=255

    # 将numpy格式转回PIL格式的原因：由于预处理transform中的方法处理的是PIL格式
    #label = Image.fromarray(label)
    plt.figure(dpi=180)  # 显示图像
    plt.imshow(label)
    plt.show()




if __name__ == "__main__":
    deal_data()