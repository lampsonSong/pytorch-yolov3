# created by lampson.song @ 2020-3-13
# Yolo layer : the detection out layer of YOLO

import torch
import torch.nn as nn

def create_grids(self, img_max_side=416, num_grids=(13, 13), device='cpu', type=torch.float32):
    nx, ny = num_grids  # x and y grid size
    self.img_max_side = img_max_side
    self.grid_stride = self.img_max_side / max(num_grids)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view((1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.grid_stride
    #self.anchor_vec = self.anchors.to(device)
    self.anchor_wh = self.anchor_vec.view(1, self.num_anchors, 1, 1, 2).to(device).type(type)
    self.grid_num_x = nx
    self.grid_num_y = ny


class YOLOLayer(nn.Module):
    # anchors is the list of anchors , e.g. [[100, 23], [45, 87], [74, 10]]
    def __init__(self, anchors, num_classes):
        super(YOLOLayer, self).__init__()

        self.anchors = anchors
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.grid_num_x = 0
        self.grid_num_y = 0
        self.num_outputs = self.num_classes + 5 # classes + len([x,y,w,h,objecteness])

    def forward(self, x, img_max_side):
        batch_size, _, grid_num_y, grid_num_x = x.shape # last layer output shape : [batch_size, 255, w, h]
        
        if (self.grid_num_x, self.grid_num_y) != (grid_num_x, grid_num_y):
            create_grids(self, img_max_side, (grid_num_x, grid_num_y), x.device, x.dtype)

        x = x.view(batch_size, self.num_anchors, self.num_outputs, self.grid_num_y, self.grid_num_x).permute(0,1,3,4,2).contiguous() # from [batch_size, 255, y, x] -> [batch_size, 3(number of anchors), y,x, 85]

        # self.training is member of nn.Module
        if self.training:
            return x
        else: # testing
            outputs = x.clone()

            # location
            outputs[:, :, :, :, :2] = torch.sigmoid(outputs[:, :, :, :, :2]) + self.grid_xy
            outputs[:, :, :, :, 2:4] = torch.exp(outputs[:,:,:,:,2:4]) * self.anchor_wh
            outputs[:,:,:,:,:4] *= self.grid_stride

            # classificaion
            torch.sigmoid_(outputs[:,:,:,:,4:])

            # from [batch_size, num_anchors, nx, ny, num_outputs] to [batch_size, -1, num_outputs]
            return outputs.view(batch_size, -1, self.num_outputs)
