import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class LayerConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LayerConv, self).__init__()
        self.conv1 = BasicConv(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = BasicConv(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UpLayerConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpLayerConv, self).__init__()

        self.conv1 = BasicConv(in_channels, out_channels, kernel_size=3)
        self.conv2 = BasicConv(out_channels, out_channels, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

def copy_crop(up_sampled, bypass, flag_crop=False):
    if flag_crop:
        c = (bypass.size()[2] - up_sampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))
    return torch.cat((up_sampled, bypass), 1)


class Unet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Unet, self).__init__()
        self.first_conv = LayerConv(in_channels=n_channels, out_channels=64)
        self.second_conv = LayerConv(in_channels=64, out_channels=128)
        self.third_conv = LayerConv(in_channels=128, out_channels=256)
        self.forth_conv = LayerConv(in_channels=256, out_channels=512)
        self.fifth_conv = LayerConv(in_channels=512, out_channels=1024)
        self.forth_out_conv = LayerConv(in_channels=1024, out_channels=512)
        self.third_out_conv = LayerConv(in_channels=512, out_channels=256)
        self.second_out_conv = LayerConv(in_channels=256, out_channels=128)
        self.first_out_conv = LayerConv(in_channels=128, out_channels=64)
        self.max_pooling = nn.MaxPool2d(kernel_size=2)
        self.up_conv4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3,
                                          stride=2, padding=1, output_padding=1)
        self.up_conv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3,
                                          stride=2, padding=1, output_padding=1)
        self.up_conv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3,
                                           stride=2, padding=1, output_padding=1)
        self.up_conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3,
                                           stride=2, padding=1, output_padding=1)

        self.final_conv = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)

    def forward(self, x):
        first_feature = self.first_conv(x)
        first_pool_feature = self.max_pooling(first_feature)

        second_feature = self.second_conv(first_pool_feature)
        second_pool_feature = self.max_pooling(second_feature)

        third_feature = self.third_conv(second_pool_feature)
        third_pool_feature = self.max_pooling(third_feature)

        forth_feature = self.forth_conv(third_pool_feature)
        forth_pool_feature = self.max_pooling(forth_feature)

        final_feature = self.fifth_conv(forth_pool_feature)

        forth_up_feature = self.up_conv4(final_feature)
        forth_cat_feature = copy_crop(forth_up_feature, forth_feature, flag_crop=True)
        forth_out_feature = self.forth_out_conv(forth_cat_feature)

        third_up_feature = self.up_conv3(forth_out_feature)
        third_cat_feature = copy_crop(third_up_feature, third_feature, flag_crop=True)
        third_out_feature = self.third_out_conv(third_cat_feature)

        second_up_feature = self.up_conv2(third_out_feature)
        second_cat_feature = copy_crop(second_up_feature, second_feature, flag_crop=True)
        second_out_feature = self.second_out_conv(second_cat_feature)

        first_up_feature = self.up_conv1(second_out_feature)
        first_cat_feature = copy_crop(first_up_feature, first_feature, flag_crop=True)
        first_out_feature = self.first_out_conv(first_cat_feature)

        output_feature = self.final_conv(first_out_feature)
        return output_feature


if __name__ == '__main__':
    net = Unet(1,1)
    x = torch.randn(1, 1, 572, 572)
    y = net(x)
    print(y.size())
