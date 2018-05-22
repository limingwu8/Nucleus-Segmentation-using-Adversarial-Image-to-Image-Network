import torch.nn as nn
from utils import Opt
import torch
import torch.nn.functional as F


def up_block(in_channels, out_channels, kernel_size=4, stride=1, padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

def up_output(in_channels, out_channels, kernel_size=4, stride=1, padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.Tanh()
    )

def down_block(in_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2)
    )

def down_output(in_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.Sigmoid()
    )

def encoding(in_channels, out_channels, kernel_size, stride, padding, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(128, out_channels, kernel_size, stride, padding),
        nn.ReLU()
    )

def left(input_res, num_features):
    out_res = nn.Sequential(
        nn.Conv2d(input_res)
    )

def transformation():
    pass

def decoding():
    pass

def conv_bn_leru(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
    )

def down_pooling():
    return nn.MaxPool2d(2)

def up_pooling(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class netG(nn.Module):
    def __init__(self, input_channels=3, nclasses=1):
        super().__init__()
        # go down
        self.conv1 = conv_bn_leru(input_channels,16)
        self.conv2 = conv_bn_leru(16, 32)
        self.conv3 = conv_bn_leru(32, 64)
        self.conv4 = conv_bn_leru(64, 128)
        self.conv5 = conv_bn_leru(128, 256)
        self.down_pooling = nn.MaxPool2d(2)

        # go up
        self.up_pool6 = up_pooling(256, 128)
        self.conv6 = conv_bn_leru(256, 128)
        self.up_pool7 = up_pooling(128, 64)
        self.conv7 = conv_bn_leru(128, 64)
        self.up_pool8 = up_pooling(64, 32)
        self.conv8 = conv_bn_leru(64, 32)
        self.up_pool9 = up_pooling(32, 16)
        self.conv9 = conv_bn_leru(32, 16)

        self.conv10 = nn.Conv2d(16, nclasses, 1)


        # test weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        # normalize input data
        x = x/255.
        # go down
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)
        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)

        # go up
        p6 = self.up_pool6(x5)
        x6 = torch.cat([p6, x4], dim=1)
        x6 = self.conv6(x6)

        p7 = self.up_pool7(x6)
        x7 = torch.cat([p7, x3], dim=1)
        x7 = self.conv7(x7)

        p8 = self.up_pool8(x7)
        x8 = torch.cat([p8, x2], dim=1)
        x8 = self.conv8(x8)

        p9 = self.up_pool9(x8)
        x9 = torch.cat([p9, x1], dim=1)
        x9 = self.conv9(x9)


        output = self.conv10(x9)
        output = F.sigmoid(output)

        return output

class netD(nn.Module):
    def __init__(self):
        super(netD, self).__init__()
        self.down_block1 = down_block(Opt.nc, 64, 4, 2, 1)
        self.down_block2 = down_block(64, 128, 4, 2, 1)
        self.down_block3 = down_block(128, 256, 4, 2, 1)
        self.down_block4 = down_block(256, 512, 4, 2, 1)
        self.down_block5 = down_block(512, 1024, 4, 2, 1)
        self.down_block6 = down_output(1024, 1, 4, 1, 0)

    def forward(self, input):
        x1 = self.down_block1(input)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)
        x6 = self.down_block6(x5)
        output = x6.view(-1, 1).squeeze(-1)
        return output
