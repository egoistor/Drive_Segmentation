import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
from PIL import Image
from data_aug import random_gamma_correction
from data_aug import random_vessel_augmentation
import numpy as np
import os

class DRIVE_Dataset(Dataset):
    def __init__(self,data_path):
        self.data_path = data_path
        self.image_path = glob.glob(os.path.join(data_path,'images/*.tif'))#TODO 更改路径
        self.manual_path = glob.glob(os.path.join(data_path,'1st_manual/*.gif'))
        self.roi_path = glob.glob(os.path.join(data_path,'mask/*.gif'))

    def augment(self,image,mode):#TODO 数据增强部分
        """
        :param image:
        :param mode: 1 :水平翻转 0 : 垂直翻转 -1 水平+垂直翻转
        :return:
        """
        # file = cv2.flip(image,mode)
        """
        :param image:
        :param mode: 1 :gamma_correction 0 : 不变 -1 vessel_augmentation
        :return:
        """
        if mode == -1:
            image = random_gamma_correction(image)
        if mode == 1:
            image = random_vessel_augmentation(image)
        return image

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self,index):
        image_path = self.image_path[index]
        manual_path = self.manual_path[index]
        roi_path = self.roi_path[index]


        #读取
        image = cv2.imread(image_path,-1)

        manual = Image.open(manual_path).convert('L')
        roi_mask = Image.open(roi_path).convert('L')

        roi_mask = 255 - np.array(roi_mask)
        label = np.clip(manual + roi_mask, a_min=0, a_max=255)
        #转为灰度图
        # plt.figure(dpi=180)  # 显示图像
        # plt.imshow(image)
        # plt.show()
        # plt.imshow(label)
        # plt.show()

        mode = random.choice([-1, 0, 1])
        # image = self.augment(image, mode)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #label = self.augment(label, mode)


        image = image.reshape(1,image.shape[0],image.shape[1])
        label = label.reshape(1,label.shape[0],label.shape[1])
        label = label/255
        #标签二值化 ，将255 -> 1
        # 随机进行数据增强,2时不做数据增强

        return image, label #TODO 不一定好使

if __name__ == "__main__":
    drive = DRIVE_Dataset("DRIVE/training/")
    print(len(drive))
    train_loader = torch.utils.data.DataLoader(drive,
                                                    batch_size=2,
                                                    shuffle=True)
    for image, label in train_loader:
        pass
    #     image = image[1:]
    #     label = label[1:]

