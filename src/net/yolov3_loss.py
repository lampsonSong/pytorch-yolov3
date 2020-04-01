# created by lampson.song @ 2020-3-23
# for loss computation

import torch
import torch.nn as nn

import sys
sys.path.append('./')

from net.yolo_layer import YOLOLayer
from net.yolov3_spp import YOLOv3_SPP

from utils.utils import wh_iou, box_giou
import numpy as np

def get_yolo_loss(model, predictions, targets, regression_loss_type='GIoU'):
    if not len(targets):
        return "NoLabels"
    data_type = torch.cuda.FloatTensor if predictions[0].is_cuda else torch.Tensor

    b_multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

    yolo_layer_idx = 0
    reg_loss, obj_loss, cls_loss = data_type([0]), data_type([0]), data_type([0])
    for idx, module in enumerate(model.module_list):
        if isinstance(module, YOLOLayer):
            yolo_out = predictions[yolo_layer_idx]
            yolo_layer_idx += 1

            feature_targets = convert_targets_to_feature_level(module, targets, train_iou_thresh=0.225)

            # box regression loss
            reg_iou_loss, reg_iou = regression_loss(yolo_out, feature_targets, regression_loss_type)
            reg_loss += reg_iou_loss

            # objectness loss
            obj_loss += objectness_loss(yolo_out, feature_targets, reg_iou, model.iou_ratio, obj_loss_type='BCE')

            # cls loss
            cls_loss += classification_loss(yolo_out, feature_targets, cls_loss_type='BCE')


    # gain are from https://github.com/ultralytics/yolov3/issues/310 
    reg_loss *= 3.54 # giou loss gain
    obj_loss *= 64.3 # obj loss gain
    cls_loss *= 37.4 # cls loss gain

    loss = reg_loss + obj_loss + cls_loss
    return loss, torch.cat((reg_loss, obj_loss, cls_loss, loss)).detach()


# loss of objectness for each grid
def objectness_loss(yolo_out, feature_targets, reg_iou, iou_ratio, obj_loss_type='BCE'):
    if obj_loss_type == 'BCE':
        obj_func = torch.nn.BCEWithLogitsLoss(reduction='mean')

    labels_obj = torch.zeros_like(yolo_out[..., 0])
    img_ids, used_anchor_idx, targets_cell_x, targets_cell_y = feature_targets['indices']

    labels_obj[img_ids, used_anchor_idx, targets_cell_x, targets_cell_y] = (1.0 - iou_ratio) + iou_ratio * reg_iou.detach().clamp(0).type(labels_obj.dtype)

    return obj_func(yolo_out[...,4], labels_obj)

# loss of classes
def classification_loss(yolo_out, feature_targets, cls_loss_type='BCE'):
    if cls_loss_type == 'BCE':
        cls_func = torch.nn.BCEWithLogitsLoss(reduction='mean')

    img_ids, used_anchor_idx, targets_cell_x, targets_cell_y = feature_targets['indices']
    # select_out shape is [num_real_targets, 85]
    select_out = yolo_out[img_ids, used_anchor_idx, targets_cell_x, targets_cell_y]
    
    labels_cls = torch.zeros_like(select_out[:,5:])

    tcls = feature_targets['tcls']
    labels_cls[range(len(tcls)), tcls] = 1.

    return cls_func(select_out[:,5:], labels_cls)

# regression loss of boxes
def regression_loss(yolo_out, feature_targets, regression_loss_type, red='mean'):
    #print(" - yolo_out is cuda : ", yolo_out.is_cuda)
    data_type = torch.cuda.FloatTensor if yolo_out.is_cuda else torch.Tensor
    reg_loss = data_type(0)

    img_ids, used_anchor_idx, targets_cell_x, targets_cell_y = feature_targets['indices']
    # select_out shape is [num_real_targets, 85]
    select_out = yolo_out[img_ids, used_anchor_idx, targets_cell_x, targets_cell_y]

    select_boxes_xy = torch.sigmoid(select_out[:,:2]) + torch.cat( (targets_cell_x.unsqueeze(1),targets_cell_y.unsqueeze(1)), 1) # sig(tx) + cx & sig(ty) + cy
    select_boxes_wh = torch.exp(select_out[:,2:4]) * feature_targets['used_anchor_vec']
    select_boxes = torch.cat((select_boxes_xy, select_boxes_wh), 1)

    tboxes = feature_targets['tboxes']

    if regression_loss_type == 'GIoU':
        reg_iou = box_giou(select_boxes, tboxes, box_type='cxcywh')
        reg_iou_loss = (1. - reg_iou).sum() if red == 'sum' else (1. - reg_iou).mean()

    return reg_iou_loss, reg_iou

# convert labels from normalized format to feature level corresponding to anchors
def convert_targets_to_feature_level(yolo_layer, targets, train_iou_thresh=0.):
    '''
    let out_num = real_targets_num * used_anchor_num
    
    output : targets_boxes[ out_num, 4] -> lx,ly,w,h
            targets_cls[ out_num, 1]
            used_anchor_vec[ out_num, 2]
            indices[out_num, out_num, out_num, out_num] -> img_ids, used_anchor_idx, cell_x, cell_y
    '''
    feature_targets = {}
    num_targets = len(targets)

    grid_num_x, grid_num_y = yolo_layer.grid_num_x, yolo_layer.grid_num_y
    anchor_vec = yolo_layer.anchor_vec
    num_anchors = len(anchor_vec)

    b_reject, b_use_all_anchors = True, True
    real_targets = targets.clone()

    # scale targets to feature map level
    real_targets[:,2:6] = real_targets[:,2:6] * torch.tensor([grid_num_x, grid_num_y, grid_num_x, grid_num_y])

    # compute iou between targets and the corresponding anchors
    targets_wh = real_targets[:,4:6]
    
    # anchor_target_iou.shape = [num_anchors, num_targets]
    anchor_target_iou = wh_iou(anchor_vec, targets_wh)

    if b_use_all_anchors:
        used_anchor_idx = torch.arange(num_anchors).view(-1,1).repeat(1, num_targets).view(-1)
        # three anchors are used for each target, so repeat three times
        real_targets = real_targets.repeat(num_anchors, 1)
    else:
        # one anchor is used for each target, targets' shape remains 
        anchor_target_iou, used_anchor_idx = anchor_target_iou.max(0)

    if b_reject:
        iou_filter = anchor_target_iou.view(-1) > train_iou_thresh
        real_targets = real_targets[iou_filter]
        used_anchor_idx = used_anchor_idx[iou_filter]

    # combine indices, convet data type to 64bit long int
    img_ids, cls, targets_cell_x, targets_cell_y = real_targets[:,:4].t().long()

    feature_targets['tboxes'] = real_targets[:, 2:6]
    feature_targets['indices'] = (img_ids, used_anchor_idx, targets_cell_x, targets_cell_y)
    feature_targets['used_anchor_vec'] = anchor_vec[used_anchor_idx] 
    feature_targets['tcls'] = cls

    return feature_targets


if __name__ == '__main__':
    # training
    yolov3_spp = YOLOv3_SPP(num_classes = 80)
    #yolov3_spp.cuda()
    input_data = torch.randn(2,3,416,416)
    #input_data.cuda()
    outputs = yolov3_spp(input_data)
   
    #targets = torch.tensor([])
    targets = torch.tensor([
        [0,20,0.2,0.3,0.1,0.2],
        [0,4,0.1,0.5,0.1,0.2],
        [1,2,0.2,0.3,0.111,0.12],
        [1,17,0.2,0.53,0.1,0.42]
        ])

    get_yolo_loss(yolov3_spp, outputs, targets)
