# created by lampson.song @ 2020-3-13
# yolov3-spp network

from net.darknet import DarkNet53
from net.darknet import ConvBnLeakyRelu
from net.darknet import weightedFeatureFusion
import torch
import torch.nn as nn
from net.yolo_layer import YOLOLayer

anchors = torch.tensor([[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90], [156,198], [373,326]])

def Conv(input_channels, output_channels, kernel_size =3, stride=1, groups=1, bias=False):
    padding = (kernel_size - 1) // 2

    seq = nn.Sequential()
    seq.add_module('Conv2d',
        nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, groups=groups, bias=bias)
    )

    return seq

class RouteConcat(nn.Module):
    def __init__(self, layers_idxes):
        super(RouteConcat, self).__init__()
        self.layers_idxes = layers_idxes
        self.n = len(layers_idxes) # number of layers

    def forward(self, outputs):
        out = torch.cat([outputs[i] for i in self.layers_idxes], 1)

        return out


class YOLOv3_SPP(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv3_SPP, self).__init__()
        self.num_classes = num_classes

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
