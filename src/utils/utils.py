# created by lampson.song @ 2020-3-17
# utils methods

import torch
import torchvision

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=img_shape[1])  # clip x
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=img_shape[0])  # clip y



def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, multi_label=True, classes=None, agnostic=False):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, conf, class)
    """
    # NMS methods https://github.com/ultralytics/yolov3/issues/679 'or', 'and', 'merge', 'vision', 'vision_batch'

    # Box constraints
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height

    method = 'vision_batch'
    batched = 'batch' in method  # run once per image, all classes simultaneously
    nc = prediction[0].shape[1] - 5  # number of classes
    multi_label &= nc > 1  # multiple labels per box
    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):
        # Apply conf constraint
        pred = pred[pred[:, 4] > conf_thres]

        # Apply width-height constraint
        pred = pred[((pred[:, 2:4] > min_wh) & (pred[:, 2:4] < max_wh)).all(1)]

        # If none remain process next image
        if not pred.shape[0]:
            continue

        # Compute conf
        pred[..., 5:] *= pred[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = cxcywh_2_x1y1x2y2(pred[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (pred[:, 5:] > conf_thres).nonzero().t()
            pred = torch.cat((box[i], pred[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        else:  # best class only
            conf, j = pred[:, 5:].max(1)
            pred = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)

        # Filter by class
        if classes:
            pred = pred[(j.view(-1, 1) == torch.tensor(classes, device=j.device)).any(1)]

        # Apply finite constraint
        if not torch.isfinite(pred).all():
            pred = pred[torch.isfinite(pred).all(1)]

        # If none remain process next image
        if not pred.shape[0]:
            continue

        # Sort by confidence
        if not method.startswith('vision'):
            pred = pred[pred[:, 4].argsort(descending=True)]

        # Batched NMS
        if batched:
            c = pred[:, 5] * 0 if agnostic else pred[:, 5]  # class-agnostic NMS
            boxes, scores = pred[:, :4].clone(), pred[:, 4]
            if method == 'vision_batch':
                i = torchvision.ops.boxes.batched_nms(boxes, scores, c, iou_thres)
            elif method == 'fast_batch':  # FastNMS from https://github.com/dbolya/yolact
                boxes += c.view(-1, 1) * max_wh
                iou = box_iou(boxes, boxes).triu_(diagonal=1)  # upper triangular iou matrix
                i = iou.max(dim=0)[0] < iou_thres

            output[image_i] = pred[i]
            continue

        # All other NMS methods
        det_max = []
        cls = pred[:, -1]
        for c in cls.unique():
            dc = pred[cls == c]  # select class c
            n = len(dc)
            if n == 1:
                det_max.append(dc)  # No NMS required if only 1 prediction
                continue
            elif n > 500:
                dc = dc[:500]  # limit to first 500 boxes: https://github.com/ultralytics/yolov3/issues/117

            if method == 'vision':
                det_max.append(dc[torchvision.ops.boxes.nms(dc[:, :4], dc[:, 4], iou_thres)])

            elif method == 'or':  # default
                # METHOD1
                # ind = list(range(len(dc)))
                # while len(ind):
                # j = ind[0]
                # det_max.append(dc[j:j + 1])  # save highest conf detection
                # reject = (bbox_iou(dc[j], dc[ind]) > iou_thres).nonzero()
                # [ind.pop(i) for i in reversed(reject)]

                # METHOD2
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:][iou < iou_thres]  # remove ious > threshold

            elif method == 'and':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < iou_thres]  # remove ious > threshold

            elif method == 'merge':  # weighted mixture box
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    i = bbox_iou(dc[0], dc) > iou_thres  # iou with other boxes
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]

            elif method == 'soft':  # soft-NMS https://arxiv.org/abs/1704.04503
                sigma = 0.5  # soft-nms sigma parameter
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    det_max.append(dc[:1])
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:]
                    dc[:, 4] *= torch.exp(-iou ** 2 / sigma)  # decay confidences
                    dc = dc[dc[:, 4] > conf_thres]  # https://github.com/ultralytics/yolov3/issues/362

        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort

    return output


def x1y1x2y2_2_cxcywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def cxcywh_2_x1y1x2y2(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def wh_iou(wh1, wh2):
    wh1 = wh1.unsqueeze(1)
    wh2 = wh2.unsqueeze(0)

    inter = torch.min(wh1, wh2).prod(2)

    return inter / (wh1.prod(2) + wh2.prod(2) - inter)

def box_giou(boxes1, boxes2, box_type='cxcywh'):
    if box_type == 'cxcywh':
        boxes1 = cxcywh_2_x1y1x2y2(boxes1)
        boxes2 = cxcywh_2_x1y1x2y2(boxes2)
    
    inter_area = torch.min(boxes1[:,2:4], boxes2[:,2:4]) - torch.max(boxes1[:,:2], boxes2[:,:2])
    inter_area = inter_area.clamp(0).prod(1) + 1e-10

    boxes1_area = (boxes1[:,2:4] - boxes1[:,:2]).prod(1) + 1e-10
    boxes2_area = (boxes2[:,2:4] - boxes2[:,:2]).prod(1) + 1e-10
    
    union = boxes1_area + boxes2_area - inter_area + 1e-10
    iou = inter_area / union

    convex_area = torch.max(boxes1[:,2:4], boxes2[:,2:4]) - torch.min(boxes1[:,:2], boxes2[:,:2])
    convex_area = convex_area.clamp(0).prod(1) + 1e-10

    return iou - (convex_area - union) / (convex_area)

def box_ciou(boxes1, boxes2, box_type='cxcywh'):
    if box_type == 'cxcywh':
        center_x1, center_y1, w1, h1 = boxes1[0], boxes1[1]
        center_x2, center_y2, w2, h2 = boxes2[0], boxes2[1]
        
        boxes1 = cxcywh_2_x1y1x2y2(boxes1)
        boxes2 = cxcywh_2_x1y1x2y2(boxes2)
    
    inter_area = torch.min(boxes1[:,2:4], boxes2[:,2:4]) - torch.max(boxes1[:,:2], boxes2[:,:2])
    inter_area = inter_area.clamp(0).prod(1) + 1e-10

    boxes1_area = (boxes1[:,2:4] - boxes1[:,:2]).prod(1) + 1e-10
    boxes2_area = (boxes2[:,2:4] - boxes2[:,:2]).prod(1) + 1e-10
    
    union = boxes1_area + boxes2_area - inter_area + 1e-10
    iou = inter_area / union

    v = 4 / (math.pi ** 2) * ( (torch.atan(w1/h1) - torch.atan(w2/h2))**2 )
    alpha = v / ((1-iou) + v)

    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    out_max_xy = torch.max(boxes1[:, 2:], boxes2[:,2:])
    out_min_xy = torch.min(boxes1[:, :2], boxes2[:,:2])
    outer = torch.clamp(out_max_xy - out_min_xy, min=0)
    outer_diag = outer[:,0] ** 2 + outer[:,2]**2

    ciou = 1 - iou + inter_diag / outer_diag + alpha * v

    return ciou
