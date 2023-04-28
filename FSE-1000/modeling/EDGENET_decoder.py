
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from PIL import Image



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.D4 = DecoderBlock(512+2+256, 200, 100, nn.BatchNorm2d)
        self.D3 = DecoderBlock(128+200+100, 100, 50, nn.BatchNorm2d)
        self.D2 = DecoderBlock(64+100+50, 50, 20, nn.BatchNorm2d)
        self.D1 = DecoderBlock(64+50+20, 20, 10, nn.BatchNorm2d)
        self.D0 = DecoderBlock(20+20+10, 20, 2, nn.BatchNorm2d, islast=True)
        self._init_weight()

    def forward(self, Dec_input_4, Dec_input_3, Dec_input_2, Dec_input_1, Dec_input_0):
        size0, size1, size2, size3 = Dec_input_0.shape[2], Dec_input_1.shape[2], Dec_input_2.shape[2], Dec_input_3.shape[2]

        D4_out, D4_skip = self.D4(Dec_input_4)
        D4_out_skip = torch.cat((D4_out, D4_skip), dim=1)
        D4_out_skip = F.interpolate(D4_out_skip, size=(size3,size3), mode='bilinear', align_corners=True)
        D3_in = torch.cat((Dec_input_3, D4_out_skip), dim=1)

        D3_out, D3_skip = self.D3(D3_in)
        D3_out_skip = torch.cat((D3_out, D3_skip), dim=1)
        D3_out_skip = F.interpolate(D3_out_skip, size=(size2,size2), mode='bilinear', align_corners=True)
        D2_in = torch.cat((Dec_input_2, D3_out_skip), dim=1)

        D2_out, D2_skip = self.D2(D2_in)
        D2_out_skip = torch.cat((D2_out, D2_skip), dim=1)
        D2_out_skip = F.interpolate(D2_out_skip, size=(size1,size1), mode='bilinear', align_corners=True)
        D1_in = torch.cat((Dec_input_1, D2_out_skip), dim=1)

        D1_out, D1_skip = self.D1(D1_in)
        D1_out_skip = torch.cat((D1_out, D1_skip), dim=1)
        D1_out_skip = F.interpolate(D1_out_skip, size=(size0,size0), mode='bilinear', align_corners=True)
        D0_in = torch.cat((Dec_input_0, D1_out_skip), dim=1)

        D_out, _ = self.D0(D0_in)

        return D_out

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

class DecoderBlock(nn.Module):
    def __init__(self, inChannel, midChannel, outChannel, BatchNorm, islast = False):
        super(DecoderBlock, self).__init__()
        self.islast = islast
        self.conv1 = nn.Conv2d(inChannel, midChannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(midChannel, midChannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(midChannel, outChannel, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = BatchNorm(midChannel)
        self.bn2 = BatchNorm(midChannel)
        self.bn3 = BatchNorm(outChannel)

        self.relu = nn.ReLU(inplace=True)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.25)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        decoder_skip = self.dropout1(out)

        out = self.conv2(decoder_skip)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        if self.islast:
            out = self.conv3(out)
        else:
            out = self.conv3(out)
            out = self.bn3(out)
            out = self.relu(out)
            out = self.dropout3(out)

        return out, decoder_skip

def build_decoder(BatchNorm):
    return Decoder(BatchNorm)