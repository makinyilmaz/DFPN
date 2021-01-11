
# coding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import math
import torchvision.ops.deform_conv as df


def conv_same(in_ch,out_ch,kernel_size):
    conv = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=kernel_size,padding=kernel_size//2,bias=True)
    return conv


class RDB(nn.Module):
    def __init__(self, channel):
        super(RDB, self).__init__()
        
        self.cnn1 = conv_same(channel,channel,3)
        self.cnn2 = conv_same(channel*2,channel,3)
        self.cnn3 = conv_same(channel*3,channel,3)
        self.down_conv = conv_same(channel*4,channel,1)
        
    def forward(self,x):
        c1 = F.relu(self.cnn1(x))
        c2 = F.relu(self.cnn2(torch.cat((x,c1),dim=1)))
        c3 = F.relu(self.cnn3(torch.cat((x,c1,c2),dim=1)))
        concat = torch.cat((x,c1,c2,c3),dim=1)
        out = self.down_conv(concat) + x

        return out



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        
        self.feature_ext = nn.Sequential(
            conv_same(1,64,3),
            RDB(64),
            RDB(64),
            conv_same(64, 64, 3)
        )
        
        self.bottleneck = nn.Sequential(
            conv_same(64*4, 64, 3),
            RDB(64),
            RDB(64),
            RDB(64),
            RDB(64),
            RDB(64),
            RDB(64),
            RDB(64),
            RDB(64),
            RDB(64),
            RDB(64),
            RDB(64),
            RDB(64),
            conv_same(64, 64, 3)
        )
        
        
        self.conv_offset = conv_same(64, 2*3*3*4, 3)
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_() 
        
        self.dcn1 = df.DeformConv2d(64, 64, kernel_size=3, padding=1)
        self.dcn2 = df.DeformConv2d(64, 64, kernel_size=3, padding=1)
        self.dcn3 = df.DeformConv2d(64, 64, kernel_size=3, padding=1)
        self.dcn4 = df.DeformConv2d(64, 64, kernel_size=3, padding=1)
    
        self.rec = nn.Sequential(
            conv_same(64*4, 64, 3),
            RDB(64),
            RDB(64),
            RDB(64),
            RDB(64),
            RDB(64),
            RDB(64),
            RDB(64),
            RDB(64),
            conv_same(64, 1, 3)
        )
        
    def forward(self,x):
        x1 = x[:,0:1]
        x2 = x[:,1:2]
        x3 = x[:,2:3]
        x4 = x[:,3:4]
        
        f1 = self.feature_ext(x1)
        f2 = self.feature_ext(x2)
        f3 = self.feature_ext(x3)
        f4 = self.feature_ext(x4)

        b = self.bottleneck(torch.cat([f1,f2,f3,f4], dim=1))
        offset1, offset2, offset3, offset4 = torch.chunk(self.conv_offset(b), 4, dim=1)
        d1 = self.dcn1(f1, offset1)     
        d2 = self.dcn2(f2, offset2)     
        d3 = self.dcn3(f3, offset3)     
        d4 = self.dcn4(f4, offset4)  
        att = torch.cat([d1,d2,d3,d4], dim=1)
        y_pred = self.rec(att)
        
        return y_pred, (torch.mean(offset1**2) + torch.mean(offset2**2) + torch.mean(offset3**2) + torch.mean(offset4**2))/4.
        
