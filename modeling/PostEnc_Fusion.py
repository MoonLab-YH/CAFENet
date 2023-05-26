import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, _expansion=2):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(inplanes / _expansion)
        self.expansion = _expansion
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, inplanes)
        self.bn3 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Mini_Postconv(nn.Module):
    def __init__(self, inplane=[64,64,128,256], midplane=[32,32,64,128], stride=1, dilation=1, downsample=None, _expansion=2, n_shot = 5):
        super(Mini_Postconv, self).__init__()
        self.post1 = Bottleneck(inplane[0], midplane[0], stride=stride, dilation=dilation, downsample=downsample, _expansion=_expansion)
        self.post2 = Bottleneck(inplane[1], midplane[1], stride=stride, dilation=dilation, downsample=downsample, _expansion=_expansion)
        self.post3 = Bottleneck(inplane[2], midplane[2], stride=stride, dilation=dilation, downsample=downsample, _expansion=_expansion)
        self.post4 = Bottleneck(inplane[3], midplane[3], stride=stride, dilation=dilation, downsample=downsample, _expansion=_expansion)

        self.dropout1 = nn.Dropout(0.15);
        self.dropout2 = nn.Dropout(0.25);
        self.dropout3 = nn.Dropout(0.25);
        self.dropout4 = nn.Dropout(0.25);

        self.n_shot = n_shot
        self._init_weight()

    def forward(self, E1,E2,E3,E4):
        out1 = self.post1(E1)
        out2 = self.post2(E2)
        out3 = self.post3(E3)
        out4 = self.post4(E4)

        out1 = self.dropout1(out1)
        out2 = self.dropout2(out2)
        out3 = self.dropout3(out3)
        out4 = self.dropout4(out4)

        return out1,out2,out3,out4


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class backbone1(nn.Module):
    def __init__(self):
        super(backbone1, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(20)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Bottleneck(20,10,_expansion=2)

        self._init_weight()

    def forward(self,x): # each [10,510,20,20]
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()