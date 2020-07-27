# created by lampson.song @ 2020-7-27
# yolov5s network

import sys
sys.path.append('.')
import torch
import torch.nn as nn
import collections

def ConvBnLeakyRelu(input_channels, output_channels, kernel_size = 3, stride = 1, groups = 1):
    padding = (kernel_size - 1) // 2

    seq = collections.OrderedDict()
    seq['conv'] =  nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, groups=groups, bias=False)
    seq['bn'] = nn.BatchNorm2d(output_channels, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    seq['act'] = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    return seq

# YOLOV5s
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential()
        self.model_dict = collections.OrderedDict()

        conv = ConvBnLeakyRelu(input_channels = 12, output_channels = 32, kernel_size = 3, stride = 1)
        #conv = nn.Sequential(conv)
        print(conv)

        self.model(conv)

    def forward(self, x):
        x = self.model(x)

        return x

if __name__ == "__main__":
    model = Model()
    print(model)

    input_data = torch.randn(1,12,5,5)
    out = model(input_data)

    print(out.shape)
