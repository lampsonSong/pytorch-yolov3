# created by lampson.song @ 2020-3-13
# yolov3-spp network

import sys
sys.path.append('.')
from net.darknet import DarkNet53
from net.darknet import ConvBnLeakyRelu
from net.darknet import weightedFeatureFusion
import torch
import torch.nn as nn
from net.yolo_layer import YOLOLayer
import math
import numpy as np
from pathlib import Path

anchors = torch.FloatTensor([[10.,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90], [156,198], [373,326]])

torch.set_printoptions(precision=5)

def Conv(input_channels, output_channels, kernel_size =3, stride=1, groups=1, bias=False):
    padding = (kernel_size - 1) // 2

    seq = nn.Sequential()
    seq.add_module('Conv2d',
        nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, groups=groups, bias=bias)
    )

    return seq

def load_darknet_weights(self):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(self.pretrained).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(self.pretrained, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    for i, module in enumerate(self.module_list[:cutoff]):
        if isinstance(module, nn.Sequential):
            conv = module[0]
            if isinstance(module[1], nn.BatchNorm2d):
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb
                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
                ptr += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw

class RouteConcat(nn.Module):
    def __init__(self, layers_idxes):
        super(RouteConcat, self).__init__()
        self.layers_idxes = layers_idxes
        self.n = len(layers_idxes) # number of layers

    def forward(self, outputs):
        out = torch.cat([outputs[i] for i in self.layers_idxes], 1)

        return out


class YOLOv3_SPP(nn.Module):
    def __init__(self, num_classes, pretrained):
        super(YOLOv3_SPP, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.iou_ratio = 1.

        # first half of yolov3-spp : darknet
        darknet53 = DarkNet53()

        self.module_list = nn.ModuleList()

        self.module_list.extend(darknet53.module_list)

        # second half of yolov3-spp
        l_params = [
                [1024, 512, 1], # input_channels, output_channels, kernel_size
                [512, 1024, 3],
                [1024, 512, 1]
                ]
        ## add convs
        for idx, l_param in enumerate(l_params):
            assert(len(l_param) == 3)

            self.module_list.append(
                ConvBnLeakyRelu(input_channels = l_param[0], output_channels = l_param[1], kernel_size = l_param[2], stride = 1)
                )

        ## SPP
        spp_params=[
                [5, [-2]],
                [9, [-4]],
                [13, [-1,-3,-5,-6]]
                ]
        for idx, spp_param in enumerate(spp_params):
            mp_kernel_size = spp_param[0]
            self.module_list.append(
                nn.MaxPool2d(kernel_size=mp_kernel_size, stride=1, padding=(mp_kernel_size-1)//2)
            )
            self.module_list.append(
                RouteConcat(spp_param[1])
            )

        ## head layers
        ### yololayer[0]
        l_head_params_1 = [
                [2048, 512, 1],
                [512, 1024, 3],
                [1024, 512, 1],
                [512, 1024, 3]
                ]
        for idx, l_param in enumerate(l_head_params_1):
            assert(len(l_param) == 3)

            self.module_list.append(
                ConvBnLeakyRelu(input_channels = l_param[0], output_channels = l_param[1], kernel_size = l_param[2], stride = 1)
                )
        self.module_list.append(
            Conv(1024, 255, kernel_size=1, bias=True)
        )

        #### YOLO out layer
        self.module_list.append(
                YOLOLayer(anchors=anchors[6:9],
                    num_classes=self.num_classes)
            )
        
        ### yololayer[1]
        self.module_list.append( RouteConcat([-4]) )
        self.module_list.append( ConvBnLeakyRelu(512, 256, kernel_size=1) )
        self.module_list.append( nn.Upsample(scale_factor=2.0) )
        self.module_list.append( RouteConcat([-1, 61]) )

        l_head_params_2 = [
                [768, 256, 1],
                [256, 512, 3],
                [512, 256, 1],
                [256, 512, 3],
                [512, 256, 1],
                [256, 512, 3]
                ]
        for idx, l_param in enumerate(l_head_params_2):
            assert(len(l_param) == 3)

            self.module_list.append(
                ConvBnLeakyRelu(input_channels = l_param[0], output_channels = l_param[1], kernel_size = l_param[2], stride = 1)
                )
        self.module_list.append(
            Conv(512, 255, kernel_size=1, bias=True)
        )
        #### YOLO out layer
        self.module_list.append(
                YOLOLayer(anchors=anchors[3:6],
                    num_classes=self.num_classes)
            )

        ### yololayer[2]
        self.module_list.append(RouteConcat([-4]))
        self.module_list.append( ConvBnLeakyRelu(256, 128, kernel_size=1) )
        self.module_list.append( nn.Upsample(scale_factor=2.0) )
        self.module_list.append( RouteConcat([-1, 36]) )

        l_head_params_3 = [
                [384, 128, 1],
                [128, 256, 3],
                [256, 128, 1],
                [128, 256, 3],
                [256, 128, 1],
                [128, 256, 3]
                ]
        for idx, l_param in enumerate(l_head_params_3):
            assert(len(l_param) == 3)

            self.module_list.append(
                ConvBnLeakyRelu(input_channels = l_param[0], output_channels = l_param[1], kernel_size = l_param[2], stride = 1)
                )
        self.module_list.append(
            Conv(256, 255, kernel_size=1, bias=True)
        )
        #### YOLO out layer
        self.module_list.append(
                YOLOLayer(anchors=anchors[0:3],
                    num_classes=self.num_classes)
            )

        self._initialize_weights()
        if self.pretrained:
            load_darknet_weights(self)
        

    def _initialize_weights(self):
        for idx, m in enumerate(self.module_list):
            #if isinstance(m, nn.Sequential):
            #    for s in m:
            #        if isinstance(s, nn.Conv2d):
            #            nn.init.kaiming_normal_(s.weight, mode='fan_out')
            #            if s.bias is not None:
            #                nn.init.zeros_(s.bias)
            #        elif isinstance(s, nn.BatchNorm2d):
            #            nn.init.ones_(s.weight)
            #            nn.init.zeros_(s.bias)
            #elif isinstance(m, nn.Conv2d):
            #    nn.init.kaiming_normal_(m.weight, mode='fan_out')
            #    if m.bias is not None:
            #        nn.init.zeros_(m.bias)
            #elif isinstance(m, nn.BatchNorm2d):
            #    nn.init.ones_(m.weight)
            #    nn.init.zeros_(m.bias)
            #elif isinstance(m, nn.Linear):
            #    nn.init.normal_(m.weight, 0, 0.01)
            #    if m.bias is not None:
            #        nn.init.zeros_(m.bias)
            #elif isinstance(m, YOLOLayer):
            if isinstance(m, YOLOLayer):
                # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
                p = math.log(1 / (m.num_classes - 0.99))

                ## arc is default
                b = [-4.5, p] # obj, cls

                bias = self.module_list[idx-1][0].bias.view(m.num_anchors, -1)
                bias[:, 4] += b[0] - bias[:, 4].mean() # obj
                bias[:, 5:] += b[1] - bias[:, 5:].mean() # cls

                self.module_list[idx-1][0].bias = torch.nn.Parameter(bias.view(-1))
                #print('----------------')
                #print(self.module_list[idx-1][0].bias)



    def forward(self, x):
        outputs = []
        yolo_out = []
        img_max_side = max(x.shape[-2:])

        for idx, module in enumerate(self.module_list):
            if isinstance(module, weightedFeatureFusion):
                x = module(x, outputs)
                outputs.append(x)
            elif isinstance(module, RouteConcat):
                x = module(outputs)
                outputs.append(x)
            elif isinstance(module, YOLOLayer):
                out = module(x,img_max_side)
                yolo_out.append(out)
                outputs.append(out)
            else:
                x = module(x)
                outputs.append(x)

        if self.training:
            return yolo_out
        else:
            return torch.cat(yolo_out, 1)

   
if __name__ == "__main__":
    yolov3spp = YOLOv3_SPP(num_classes=80)
    yolov3spp.eval()
    print(yolov3spp)

    input_data = torch.randn(1,3,416,416)
    out = yolov3spp(input_data)

    print(out.shape)
