# created by lampson.song @ 2020-7-8
# to create CSPDarknet53 net structure

import torch
import torch.nn as nn
import torch.nn.functional as F

class RouteConcat(nn.Module):
    def __init__(self, layers_idxes):
        super(RouteConcat, self).__init__()
        self.layers_idxes = layers_idxes
        self.n = len(layers_idxes) # number of layers

    def forward(self, outputs):
        out = torch.cat([outputs[i] for i in self.layers_idxes], 1)

        return out

class weightedFeatureFusion(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers_idxes, weight=False):
        super(weightedFeatureFusion, self).__init__()
        self.layers_idxes = layers_idxes  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers_idxes) + 1  # number of layers
        if weight:
            self.w = torch.nn.Parameter(torch.zeros(self.n))  # layer weights

    def forward(self, x, outputs):
        # Weights
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]

        # Fusion
        nc = x.shape[1]  # input channels
        for i in range(self.n - 1):
            a = outputs[self.layers_idxes[i]] * w[i + 1] if self.weight else outputs[self.layers_idxes[i]]  # feature to add
            ac = a.shape[1]  # feature channels
            dc = nc - ac  # delta channels

            # Adjust channels
            if dc > 0:  # slice input
                x[:, :ac] = x[:, :ac] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            elif dc < 0:  # slice feature
                x = x + a[:, :nc]
            else:  # same shape
                x = x + a
        return x

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

def ConvBnMish(input_channels, output_channels, kernel_size =3, stride=1, groups=1):
    padding = (kernel_size - 1) // 2

    seq = nn.Sequential()
    seq.add_module('Conv2d',
        nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, groups=groups, bias=False)
        )
    seq.add_module('BatchNorm2d',
        nn.BatchNorm2d(output_channels, momentum=0.1)
        )
    seq.add_module('activation',
        Mish()
        )

    return seq


# input_channels = output_channels, down channels then conv
def CSPDarkConvRes(num_blocks, t_sizes, m_list, kernel_sizes=[1,3]):
    assert(len(kernel_sizes) == 2)

    m_list.append(
        ConvBnMish(input_channels = t_sizes[0], output_channels = t_sizes[1], kernel_size = kernel_sizes[0], stride = 1, groups = 1)
        )
    for i in range(0,num_blocks):
        m_list.append(
            ConvBnMish(input_channels = t_sizes[1], output_channels = t_sizes[1], kernel_size = kernel_sizes[0], stride = 1, groups = 1)
            )
        m_list.append(
            ConvBnMish(input_channels = t_sizes[1], output_channels = t_sizes[1], kernel_size = kernel_sizes[1], stride = 1, groups = 1)
            )
        
        m_list.append(weightedFeatureFusion([-3], False))

    m_list.append(
            ConvBnMish(input_channels = t_sizes[1], output_channels = t_sizes[1], kernel_size = kernel_sizes[0], stride = 1, groups = 1)
            )

class CSPDarkNet53(nn.Module):
    def __init__(self):
        super(CSPDarkNet53, self).__init__()
        self.module_list = nn.ModuleList()

        # set parameters
        in_out_channels=[
                    [32, 64],
                    [64, 128],
                    [128, 256],
                    [256, 512],
                    [512, 1024]
                ]
        residual_num_blocks=[
                1,
                2,
                8,
                8,
                4
            ]
        route_idxes = [
                2, 
                12, 
                25, 
                56, 
                87
            ]

        transition_in_out_channels=[
                    [64, 64],
                    [128, 64],
                    [256, 128],
                    [512, 256],
                    [1024, 512]
                ]

        concat_in_out_channels=[
                    [128, 64],
                    [128, 128],
                    [256, 256],
                    [512, 512],
                    [1024, 1024]
                ]
        # first conv
        conv1 = ConvBnMish(input_channels = 3,output_channels = 32, kernel_size = 3, stride = 1)
        self.module_list.append(conv1)
        
        # downsample and residuals
        for idx in range(0, len(in_out_channels)):
            c_sizes = in_out_channels[idx]
            t_sizes = transition_in_out_channels[idx]
            concat_sizes = concat_in_out_channels[idx]

            ## downsample
            self.module_list.append(
                ConvBnMish(input_channels = c_sizes[0], output_channels = c_sizes[1], kernel_size = 3, stride = 2)
                )
            # transition layer
            self.module_list.append(
                ConvBnMish(input_channels = t_sizes[0], output_channels = t_sizes[1], kernel_size = 1, stride = 1)
                )
            self.module_list.append(
                RouteConcat([-2])
                )

            ## residual
            CSPDarkConvRes(num_blocks=residual_num_blocks[idx], t_sizes=t_sizes, m_list=self.module_list)

            # RouteConcat (Route)
            self.module_list.append(
                RouteConcat([-1, route_idxes[idx]])
                    )
            self.module_list.append(
                ConvBnMish(input_channels = concat_sizes[0], output_channels = concat_sizes[1], kernel_size = 1, stride = 1)
                    )

        ## for classification
        #num_classes = 1000
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)


    def forward(self, x):
        outputs = []
        for idx, module in enumerate(self.module_list):
            if(isinstance(module, weightedFeatureFusion)):
                x = module(x, outputs)
                outputs.append(x)
            elif isinstance(module, RouteConcat):
                x = module(outputs)
                outputs.append(x)
            else:
                x = module(x)
                outputs.append(x)

        return x

if __name__ == "__main__":
    body = CSPDarkNet53()
    print(body)
    input_data = torch.randn(1,3,256,256)

    out = body(input_data)

    print(out.shape)

    #torch.onnx.export(body, input_data, "CSPDarknet.onnx", verbose=True, keep_initializers_as_inputs=True)

    #m = torch.load("/home/lampson/workspace-ln/objectDetection/YOLO/ultralytics-yolov3/weights/yolov3_model/416/yolov3-spp-ultralytics.pt")
    #print("---------------")
    #params_dict = m['model']
    ##for key, value in params_dict.items():
    ##    print(key)
    ##    print(value)

    #body.load_state_dict(params_dict)
