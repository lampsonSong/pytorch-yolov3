# created by lampson.song @ 2020-7-8
# to create CSPDarknet53 net structure

import torch
import torch.nn as nn

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



def ConvBnLeakyRelu(input_channels, output_channels, kernel_size =3, stride=1, groups=1):
    padding = (kernel_size - 1) // 2

    seq = nn.Sequential()
    seq.add_module('Conv2d',
        nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, groups=groups, bias=False)
        )
    seq.add_module('BatchNorm2d',
        nn.BatchNorm2d(output_channels, momentum=0.1)
        )
    seq.add_module('activation',
        nn.LeakyReLU(0.1, inplace=True)
        )

    return seq

# input_channels = output_channels, down channels then conv
def DarkConvRes(num_blocks, input_channels, m_list, kernel_sizes=[1,3]):
    assert(len(kernel_sizes) == 2)

    for i in range(0,num_blocks):
        m_list.append(
            ConvBnLeakyRelu(input_channels = input_channels, output_channels = input_channels//2, kernel_size = kernel_sizes[0], stride = 1, groups = 1)
            )
        m_list.append(
            ConvBnLeakyRelu(input_channels = input_channels//2, output_channels = input_channels, kernel_size = kernel_sizes[1], stride = 1, groups = 1)
            )
        
        m_list.append(weightedFeatureFusion([-3], False))

class DarkNet53(nn.Module):
    # BTW, darknet-53 contains only 52 convolution layers
    def __init__(self):
        super(DarkNet53, self).__init__()
        self.module_list = nn.ModuleList()
        
        # set parameters
        in_out_channels=[
                    [32, 64],
                    [64, 128],
                    [128, 256],
                    [256, 512],
                    [512, 1024]
                ]
        residual_num_blocks=[1,2,8,8,4]
        assert(len(in_out_channels) == len(residual_num_blocks))

        # shortcut layers
        #shortcut_idx = [4, 8, 11, 15, 18, 21, 24, 27, 30, 33, 35]

        # first conv
        conv1 = ConvBnLeakyRelu(input_channels = 3,output_channels = 32, kernel_size = 3, stride = 1)
        self.module_list.append(conv1)
        
        # downsample and residuals
        for idx in range(0, len(in_out_channels)):
            c_sizes = in_out_channels[idx]
            assert(len(c_sizes) == 2)

            ## downsample
            self.module_list.append(
                ConvBnLeakyRelu(input_channels = c_sizes[0], output_channels = c_sizes[1], kernel_size = 3, stride = 2)
                )
            ## residual
            DarkConvRes(num_blocks=residual_num_blocks[idx], input_channels=c_sizes[1], m_list=self.module_list)
       
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
            else:
                x = module(x)
                outputs.append(x)

        return x

if __name__ == "__main__":
    body = DarkNet53()
    print(body)
    input_data = torch.randn(1,3,256,256)

    out = body(input_data)

    print(out.shape)

    #m = torch.load("/home/lampson/workspace-ln/objectDetection/YOLO/ultralytics-yolov3/weights/yolov3_model/416/yolov3-spp-ultralytics.pt")
    #print("---------------")
    #params_dict = m['model']
    ##for key, value in params_dict.items():
    ##    print(key)
    ##    print(value)

    #body.load_state_dict(params_dict)
