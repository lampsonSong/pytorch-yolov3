# created by lampson.song @ 2020-3-13
# yolov3-spp network

import sys
sys.path.append('.')
import torch
import torch.nn as nn

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        # softplus(x) = log(1+e^x)
        softplus_x = torch.log(1 + torch.exp(x))
        # tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        tmp = torch.exp(softplus_x)
        # out = x * tanh(softplus(x))
        return x * (tmp - 1 / tmp) / (tmp + 1 / tmp)

def ConvBnMish(input_channels, output_channels, kernel_size = 3, stride = 1, groups = 1):
    padding = (kernel_size - 1) // 2

    seq= nn.Sequential()
    seq.add_module('Conv2d', 
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, groups=groups, bias=False)
            )
    seq.add_module('BatchNorm2d',
            nn.BatchNorm2d(output_channels)
            )
    seq.add_module('Mish',
            Mish()
            )

    return seq

class YOLOV4(nn.Module):
    def __init__(self):
        super(YOLOV4, self).__init__()
        self.module_list = nn.ModuleList()

        conv1 = ConvBnMish(input_channels = 3, output_channels = 32, kernel_size = 3, stride = 1)
        self.module_list.append(conv1)

    def forward(self, x):
        for idx, module in enumerate(self.module_list):
            x = module(x)

        return x

if __name__ == "__main__":
    model = YOLOV4()
    print(model)

    input_data = torch.randn(1,3,5,5)
    out = model(input_data)

    print(out.shape)
