import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from PIL import Image

class AttMaker(nn.Module):
    def __init__(self):
        super(AttMaker, self).__init__()
        self.A4 = AttenBlock(512+2+256, 200, 100, nn.BatchNorm2d)
        self.A3 = AttenBlock(128+200+100, 100, 50, nn.BatchNorm2d)
        self.A2 = AttenBlock(64+100+50, 50, 20, nn.BatchNorm2d)
        self.A1 = AttenBlock(64+50+20, 20, 10, nn.BatchNorm2d)
        self.A0 = AttenBlock(20+20+10, 20, 2, nn.BatchNorm2d, islast=True)
        self._init_weight()

    def forward(self, Att_input_4, Att_input_3, Att_input_2, Att_input_1, Att_input_0):
        size0, size1, size2, size3 = Att_input_0.shape[2], Att_input_1.shape[2], Att_input_2.shape[2], Att_input_3.shape[2]

        A4_out, A4_skip, Att4 = self.A4(Att_input_4)
        A4_out_skip = torch.cat((A4_out, A4_skip), dim=1)
        A4_out_skip = F.interpolate(A4_out_skip, size=(size3,size3), mode='bilinear', align_corners=True)
        A3_in = torch.cat((Att_input_3, A4_out_skip), dim=1)

        A3_out, D3_skip, Att3 = self.A3(A3_in)
        A3_out_skip = torch.cat((A3_out, D3_skip), dim=1)
        A3_out_skip = F.interpolate(A3_out_skip, size=(size2,size2), mode='bilinear', align_corners=True)
        A2_in = torch.cat((Att_input_2, A3_out_skip), dim=1)

        A2_out, D2_skip, Att2 = self.A2(A2_in)
        A2_out_skip = torch.cat((A2_out, D2_skip), dim=1)
        A2_out_skip = F.interpolate(A2_out_skip, size=(size1,size1), mode='bilinear', align_corners=True)
        A1_in = torch.cat((Att_input_1, A2_out_skip), dim=1)

        A1_out, D1_skip, Att1 = self.A1(A1_in)
        A1_out_skip = torch.cat((A1_out, D1_skip), dim=1)
        A1_out_skip = F.interpolate(A1_out_skip, size=(size0,size0), mode='bilinear', align_corners=True)
        A0_in = torch.cat((Att_input_0, A1_out_skip), dim=1)

        A_out, _, Att0 = self.A0(A0_in)

        return Att4, Att3, Att2, Att1, Att0

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class AttenBlock(nn.Module):
    def __init__(self, inChannel, midChannel, outChannel, BatchNorm, islast = False):
        super(AttenBlock, self).__init__()
        self.islast = islast
        self.conv1 = nn.Conv2d(inChannel, midChannel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(midChannel, outChannel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(outChannel, 1, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm(midChannel)
        self.bn2 = BatchNorm(outChannel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        decoder_skip = self.relu(out)

        out = self.conv2(decoder_skip)
        out = self.bn2(out)
        out = self.relu(out)

        Att = self.conv3(out)

        return out, decoder_skip, Att

def build_decoder(BatchNorm):
    return AttMaker(BatchNorm)