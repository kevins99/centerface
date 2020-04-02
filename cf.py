import torch.nn as nn
from torch.nn import Sequential
import torch
import numpy as np
from torchvision import models
import torch.nn.functional as f
from matplotlib import pyplot as plt
from matplotlib import cm as CM

class centerface(nn.Module):
   
    
    def __init__(self):
        super(centerface, self).__init__()



# (self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=True): conv
        self.mobilenet_v2_final = Conv2d(320,24,1,1)

# def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, relu=True, same_padding=False, bn=True): deconv

        self.lane_0_transpose = ConvTranspose2d(24,24,2,2)
        self.lane_1_conv = Conv2d(96,24,1,1)
        #lane_0_transpose + lane_1_conv

        self.lane_0_1_transpose =  ConvTranspose2d(24,24,2,2)
        self.lane_2_conv = Conv2d(32,24,1,1)
        #lane_0_1_transpose + lane_2_conv

        self.lane_0_1_2_transpose = ConvTranspose2d(24,24,2,2)
        self.lane_3_conv = Conv2d(24,24,1,1)
        #lane_0_1_2_transpose + lane_3_conv

        self.final_merge = Conv2d(24,24,3,1,same_padding=True)


        self.hmap = nn.Conv2d(24, 1, 1, 1, padding=0,bias=True)
        self.sig_out=nn.Sigmoid()

        self.size = nn.Conv2d(24, 2, 1, 1, padding=0,bias=True)
        self.offset = nn.Conv2d(24, 2, 1, 1, padding=0,bias=True)
        self.lms = nn.Conv2d(24, 10, 1, 1, padding=0,bias=True)


        self._initialize_weights()





        mobilenet_v2 = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
        self.mobilenet_v2=nn.ModuleList(list(mobilenet_v2.features)[:-1])
        # for p in self.mobilenet_v2.parameters():
        #     p.requires_grad=False

        self.output_layer=Conv2d(32,1,1,same_padding=True,bn=False,relu=True)


        

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


##############################################################

        
    def forward(self, x):

######################################################################
        
        result=[]
        for idx,model in enumerate(self.mobilenet_v2):
            x=model(x)
            if(idx in {3,6,13,len(self.mobilenet_v2)-1}):
                result.append(x)
        lane0 = self.mobilenet_v2_final(result[-1])
        lane_0_transpose = self.lane_0_transpose(lane0)
        lane_1_conv = self.lane_1_conv(result[-2])

        lane_0_1 = lane_0_transpose + lane_1_conv

        lane_0_1_transpose = self.lane_0_1_transpose(lane_0_1)
        lane_2_conv = self.lane_2_conv(result[-3])

        lane_0_1_2 = lane_0_1_transpose + lane_2_conv

        lane_0_1_2_transpose = self.lane_0_1_2_transpose(lane_0_1_2)
        lane_3_conv = self.lane_3_conv(result[-4])

        merge = lane_0_1_2_transpose + lane_3_conv
        final_merge = self.final_merge(merge)

        hmap = self.sig_out(self.hmap(final_merge))
        size = self.size(final_merge)
        offset = self.offset(final_merge)
        # lms = self.lms(final_merge)


        # print(hmap.shape, size.shape, offset.shape)

        return hmap, size, offset







        












    
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=True):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding,bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.9, affine=True,track_running_stats=True) if bn else None
        self.relu = nn.ReLU6(inplace=False) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, relu=True, same_padding=False, bn=True):
        super(ConvTranspose2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding,bias=False,dilation=1)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.9, affine=True,track_running_stats=True) if bn else None
        self.relu = nn.ReLU6(inplace=False) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x




#################################




