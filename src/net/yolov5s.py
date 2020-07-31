# created by lampson.song @ 2020-7-27
# yolov5s network

import sys
sys.path.append('.')
import torch
import torch.nn as nn
import collections
from net.yolo_layer import YOLOLayer

anchors = torch.FloatTensor([[10.,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90], [156,198], [373,326]])

class Conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size = 3, stride = 1, groups = 1, act=True):
        super(Conv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(output_channels, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(input_channels=c1 * 4, output_channels=c2, kernel_size=k)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

class Bottleneck(nn.Module):
    def __init__(self, input_channels, output_channels, shortcut=True, groups=1, expansion=0.5):
        super(Bottleneck, self).__init__()
        ec = int(input_channels * expansion)
        self.bottle_conv1 = Conv(input_channels, ec, kernel_size=1, stride=1)
        self.bottle_conv2 = Conv(ec, output_channels, kernel_size=3, stride=1)
        self.add = shortcut and input_channels == output_channels

    def forward(self, x):
        return x + self.bottle_conv2(self.bottle_conv1(x)) if self.add else self.bottle_conv2(self.bottle_conv1(x))

class BottleneckCSP(nn.Module):
    def __init__(self, input_channels, output_channels, repeat=1, shortcut=True, groups=1, expansion=0.5, act=True):
        super(BottleneckCSP, self).__init__()
        ec = int(input_channels * expansion)
        self.conv1 = Conv(input_channels, ec, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_channels, ec, kernel_size=1, stride=1, bias=False)
        self.m = nn.Sequential(*[Bottleneck(ec, ec, shortcut, groups=groups, expansion=1.0) for _ in range(repeat)])

        self.conv3 = nn.Conv2d(ec, ec, kernel_size=1, stride=1, bias=False)
        self.conv4 = Conv(ec*2, output_channels, kernel_size=1, stride=1)

        self.bn = nn.BatchNorm2d(ec*2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True) if act else nn.Identity()
        

    def forward(self, x):
        y1 = self.conv3(self.m(self.conv1(x)))
        y2 = self.conv2(x)

        return self.conv4(self.act(self.bn(torch.cat((y1,y2), dim=1))))

class SPP(nn.Module):
    def __init__(self, input_channels, output_channels, k=(5,9,13)):
        super(SPP, self).__init__()
        c = input_channels // 2
        self.spp_conv1 = Conv(input_channels, c, kernel_size=1, stride=1)
        self.spp_conv2 = Conv(c * (len(k) +1), output_channels, kernel_size=1, stride=1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x//2) for x in k])

    def forward(self, x):
        x = self.spp_conv1(x)
        return self.spp_conv2(torch.cat([x] + [m(x) for m in self.m], dim=1))

class Concat(nn.Module):
    def __init__(self, layers_idxes):
        super(Concat, self).__init__()
        self.layers_idxes = layers_idxes

    def forward(self, outputs):
        return torch.cat([outputs[i] for i in self.layers_idxes], dim=1)

# YOLOV5s
class YOLOv5s(nn.Module):
    def __init__(self, num_classes, pretrained=None):
        super(YOLOv5s, self).__init__()

        self.module_list = nn.ModuleList()
        self.layers = [ 
                Focus(c1=3, c2=32, k=3), # 0
                Conv(32, 64), # 1
                BottleneckCSP(64,64), # 2
                Conv(64, 128, stride=2), # 3
                BottleneckCSP(128, 128, repeat=3), # 4
                Conv(128, 256, stride=2), # 5
                BottleneckCSP(256, 256, repeat=3), # 6
                Conv(256, 512, stride=2), # 7
                SPP(512, 512), # 8
                BottleneckCSP(512, 512, repeat=1), # 9
                Conv(512, 256, kernel_size=1), # 10
                nn.Upsample(scale_factor=2.0), # 11
                Concat(layers_idxes=[-1, 6]), # 12
                BottleneckCSP(512, 256, expansion=0.25), # 13
                Conv(256, 128), # 14
                nn.Upsample(scale_factor=2.0), # 15
                Concat(layers_idxes=[-1, 4]), # 16
                BottleneckCSP(256, 128, expansion=0.25), # 17
                nn.Conv2d(128, 255, kernel_size = 1, stride = 1), # 18
                YOLOLayer(anchors=anchors[0:3], num_classes=num_classes), # yolo out 1
                Concat(layers_idxes=[17]), # 19
                Conv(128, 128, stride=2), # 20
                Concat(layers_idxes=[-1, 14]), # 21
                BottleneckCSP(256, 256, expansion=0.5), # 22
                nn.Conv2d(256, 255, kernel_size = 1, stride = 1), # 23
                YOLOLayer(anchors=anchors[3:6], num_classes=num_classes), # yolo out 2
                Concat(layers_idxes=[22]), # 24
                Conv(256, 256, stride=2), # 25
                Concat(layers_idxes=[-1, 10]), # 26
                BottleneckCSP(512, 512, expansion=0.5), # 27
                nn.Conv2d(512, 255, kernel_size = 1, stride = 1), # 28
                YOLOLayer(anchors=anchors[6:9], num_classes=num_classes), # yolo out 3
                ]

        for l in self.layers:
            self.module_list.append(l)

        #self.model = nn.Sequential(*self.layers)


    def forward(self, x):
        outputs = []
        yolo_out = []
        img_max_side = max(x.shape[-2:])
        
        for m in self.module_list:
            if isinstance(m, Concat):
                x = m(outputs)
                outputs.append(x)
            elif isinstance(m, YOLOLayer):
                out = m(x, img_max_side)
                yolo_out.append(out)
            else:
                x = m(x)
                outputs.append(x)

        if self.training:
            return yolo_out
        else:
            return torch.cat(yolo_out, 1)

if __name__ == "__main__":
    model = YOLOv5s(num_classes=80)
    print(model)

    input_data = torch.randn(1,3,128,128)
    out = model(input_data)

    print(out.shape)
