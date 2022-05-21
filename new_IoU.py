import cv2
import os
import numpy as np
from pathlib import Path
import time
from PIL import Image
import numpy as np

def compute_mIoU():
    # Read images
    sum_IoU = 0
    for ind in range(29):  # 读取每一个（图片-标签）对
        pred_imgs_str = 'result/' + str(ind) + '_res.png'
        pred_img = np.array(Image.open(pred_imgs_str))
        print(pred_img)
        gt_imgs_str = 'data/test/' + str(ind) + '.png'
        gt_img = np.array(Image.open(gt_imgs_str))
        print(gt_img)
        intersection = np.sum(np.logical_and(pred_img, gt_img[:, 0]))
        union = np.sum(np.logical_or(pred_img, gt_img[:, 0]))
        sum_IoU = sum_IoU + intersection / union

    iou_score = sum_IoU/29
    return  iou_score

if __name__ == "__main__":
    mIoU = compute_mIoU()
    print("图片的mIoU为：",mIoU)