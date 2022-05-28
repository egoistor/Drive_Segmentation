'''
Author: weifeng liu
Date: 2022-04-13 22:44:33
LastEditTime: 2022-04-14 14:29:34
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /BIBM-project/segmentation_pipeline/loss/dice.py
'''
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Args:
            inputs (tensor): model outputs
            targets (tensor): image labels
            smooth (int, optional): smooth factor. Defaults to 1.

        Returns:
            loss
        """
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class calc_loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(calc_loss, self).__init__()

    def forward(self, prediction, target, bce_weight=0.5):
        """Calculating the loss and metrics
        Args:
            prediction = predicted image
            target = Targeted image
            metrics = Metrics printed
            bce_weight = 0.5 (default)
        Output:
            loss : dice loss of the epoch """
        bce = F.binary_cross_entropy_with_logits(prediction, target)
        prediction = F.sigmoid(prediction)
        dice = self.dice_loss(prediction, target)

        loss = bce * bce_weight + dice * (1 - bce_weight)

        return loss

    def dice_loss(self, prediction, target):
        """Calculating the dice loss
        Args:
            prediction = predicted image
            target = Targeted image
        Output:
            dice_loss"""

        smooth = 1.0

        i_flat = prediction.view(-1)
        t_flat = target.view(-1)

        intersection = (i_flat * t_flat).sum()

        return 1 - ((2. * intersection + smooth) /
                    (i_flat.sum() + t_flat.sum() + smooth))


def threshold_predictions_v(predictions, thr=150):
    thresholded_preds = predictions[:]
    # hist = cv2.calcHist([predictions], [0], None, [2], [0, 2])
    # plt.plot(hist)
    # plt.xlim([0, 2])
    # plt.show()
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 255
    return thresholded_preds


def threshold_predictions_p(predictions, thr=0.01):
    thresholded_preds = predictions[:]
    #hist = cv2.calcHist([predictions], [0], None, [256], [0, 256])
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds


if __name__ == '__main__':
    from dataset_DRIVE import DRIVE_Dataset
    import pdb
    data_path = "DRIVE/training/"
    batch_size = 1
    criterion = DiceLoss()
    drive_dataset = DRIVE_Dataset(data_path)
    train_loader = torch.utils.data.DataLoader(drive_dataset,
                                               batch_size,
                                               shuffle=True)
    for images, labels in train_loader:
        # pdb.set_trace()
        loss = criterion(labels, labels)
        print(loss)
