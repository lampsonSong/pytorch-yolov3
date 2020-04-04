# created by lampson.song @ 2020-3-3
# read coco for pytorch

import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
#import cv2

from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import random
import cv2

import imgaug as ia
import imgaug.augmenters as iaa

import sys
sys.path.append('./')
from dataset.coco_dataset import COCODataset
from utils.utils import x1y1x2y2_2_cxcywh
from utils.utils import cxcywh_2_x1y1x2y2
from utils.utils import scale_coords

# from https://github.com/ultralytics/yolov3 utils/dataset.py
def letterbox(img, new_shape=(416, 416), color=(128, 128, 128),
              auto=True, scaleFill=False, scaleup=True, interp=cv2.INTER_LINEAR):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = max(new_shape) / max(shape)
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=interp)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    return img, ratio, (left, right, top, bottom)


class COCODatasetYolo(COCODataset):
    def __init__(self, coco_dir, set_name='val2014', img_size=416, multiscale=False, phase='Test'):
        super().__init__(coco_dir, set_name, img_size)

        self.max_objects = 100
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32

        self.multiscale = multiscale
        self.batch_count = 0
        self.augmentation = False

        if phase == 'Train':
            self.augmentation = True

            self.aug_seq = iaa.Sequential([
                iaa.Affine(
                    scale=(0.8, 1.2),
                    rotate=(-10,10),
                    shear=(-15,15),
                    translate_percent=(-0.05,0.05),
                    ),
                iaa.Fliplr(p=0.5),
                iaa.MultiplyHueAndSaturation((0.5,1.5), per_channel=True)
                ])

    def __getitem__(self, index):
        lettered_img, ratio, pad = self.load_image(index)
        targets, ia_boxes = self.load_box_annotation_yolo(index, ratio, pad)
        #print("-- targets : ", targets)

        if self.augmentation:
            lettered_img, aug_ia_boxes = self.aug_seq(image=lettered_img, bounding_boxes=ia_boxes)

            for i,aug_ia_box in enumerate(aug_ia_boxes):
                #print(i," - aug_ia_box : ", aug_ia_box)
                targets[i][2], targets[i][3] = torch.tensor(aug_ia_box.x1), torch.tensor(aug_ia_box.y1)
                targets[i][4], targets[i][5] = torch.tensor(aug_ia_box.x2), torch.tensor(aug_ia_box.y2)

        # label from [x1, y1, x2, y2] to [cx, cy, w, h] and noralization
        lettered_img_shape = torch.tensor(lettered_img.shape[:2]).repeat(1,2).unsqueeze(0)
        targets[:,2:6] = x1y1x2y2_2_cxcywh(targets[:,2:6]) / lettered_img_shape

        # filter the (cx, cy) out of index
        t_in = (targets[:,2] > 0.) * (targets[:,2] < 1.) * (targets[:,3] > 0.) * (targets[:,3] < 1.)

        targets = targets[t_in, :]

        
        # BGR2RGB
        lettered_img = cv2.cvtColor(lettered_img, cv2.COLOR_BGR2RGB)
        lettered_img = torch.from_numpy(lettered_img)
        # from [w,h,c] to [1,c,w,h]
        lettered_img = lettered_img.permute(2,0,1)


        return lettered_img, targets

    def load_image(self, index):
        img_info = self.coco.loadImgs(self.image_ids[index])[0]
        img_path = os.path.join(self.coco_dir, 'images', self.set_name, img_info['file_name'])
       
        img = cv2.imread(img_path)
        
        # resize image keeping ratio
        lettered_img, ratio, pad = letterbox(img, (self.img_size, self.img_size), auto=False)

        # Handle images with less than three channels
        if len(lettered_img.shape) != 3:
            lettered_img = lettered_img.unsqueeze(0)
            lettered_img = lettered_img.expand((3, img.shape[1:]))

        return lettered_img, ratio, pad


    def load_box_annotation_yolo(self, index, ratio, pad):
        boxes = self.load_box_annotation(index)
        
        # Extract coordinates for unpadded + unscaled image
        x1 = boxes[:,2]
        y1 = boxes[:,3]
        w = boxes[:,4]
        h = boxes[:,5]

        # Returns (x1, y1, x2, y2)
        boxes[:, 2] = ratio[0] * x1 + pad[0]
        boxes[:, 3] = ratio[1] * y1 + pad[2]
        boxes[:, 4] = ratio[0] * w + boxes[:, 2]
        boxes[:, 5] = ratio[1] * h + boxes[:, 3]

        ia_boxes = []
        if self.augmentation:
            ia_boxes = ia.BoundingBoxesOnImage([
                ia.BoundingBox(x1=box[2], x2=box[4], y1=box[3], y2=box[5]) for box in boxes
                ], shape=(self.img_size, self.img_size))

        return boxes, ia_boxes


    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        #print("== imgs : ", imgs)
        #print("== targets : ", targets)
        
        # remove empty placeholder targets, from tuple to list
        targets = [boxes for boxes in targets if boxes is not None]
        # add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
    
        # select new image size every tenth step
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        # stack images
        imgs = torch.stack([img for img in imgs])
        self.batch_count += 1
        return imgs, targets



# sample of calling

if __name__ == "__main__":
    coco_dir = "/home/lampson/2T_disk/Data/COCO_CommonObjectsInContext/COCO_2014"
    dataset = "val2014"
    
    test_dataset = COCODatasetYolo(coco_dir, dataset, img_size=608, phase='Train')
    test_params = {
            "batch_size" : 1,
            "shuffle" : False,
            "drop_last" : False,
            "num_workers" : 0,
            "collate_fn" : test_dataset.collate_fn
            }
    test_generator = DataLoader(test_dataset, **test_params)
    
    for i, (imgs, targets) in enumerate(test_generator):
        #print("imgs : ", imgs.shape)
        #print("targets : ", targets)
        img = imgs[0,:,:,:]
        
        # from tensor to numpy
        im = img.numpy().transpose(1,2,0)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
       
        # from normalized [cx,cy,w,h] to [x1, y1, x2, y2]
        im_shape = torch.tensor(im.shape[:2]).repeat(1,2).unsqueeze(0)
        targets[:,2:6] = cxcywh_2_x1y1x2y2(targets[:,2:6]) * im_shape

        for _ , target in enumerate(targets):
            cv2.rectangle(im, (int(target[2]), int(target[3])), (int(target[4]), int(target[5])), (0,255,0))

        cv2.imshow("demo", im)
        cv2.waitKey(0)
        
