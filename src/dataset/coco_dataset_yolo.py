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

from coco_dataset import COCODataset

def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)  
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0) 
    # Add padding    
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad  

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class COCODatasetYolo(COCODataset):
    def __init__(self, coco_dir, set_name='val2014', img_size=416):
        super().__init__(coco_dir, set_name, img_size)
        self.augment=True
        self.multiscale=True
        self.normalized_labels=True

        self.max_objects = 100
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32

    def __getitem__(self, index):
        img, pad, padded_h, padded_w = self.load_image(index)
        targets = self.load_box_annotation_yolo(index, pad, padded_h, padded_w)

        return img, targets

    def load_image(self, index):
        img_info = self.coco.loadImgs(self.image_ids[index])[0]
        img_path = os.path.join(self.coco_dir, 'images', self.set_name, img_info['file_name'])
       
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        # the image size original
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        return img, pad, padded_h, padded_w


    def load_box_annotation_yolo(self, index, pad, padded_h, padded_w):
        boxes = self.load_box_annotation(index)
        
        # Extract coordinates for unpadded + unscaled image
        x1 = boxes[:,2]
        y1 = boxes[:,3]
        x2 = boxes[:,2] + boxes[:,4]
        y2 = boxes[:,3] + boxes[:,5]
        # Adjust for added padding
        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[1]
        y2 += pad[3]
        # Returns (cx, cy, w, h)
        boxes[:, 2] = ((x1 + x2) / 2) / padded_w
        boxes[:, 3] = ((y1 + y2) / 2) / padded_h
        boxes[:, 4] /= padded_w
        boxes[:, 5] /= padded_h

        return boxes


    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        #print("== targets : ", targets)
        
        # remove empty placeholder targets, from tuple to list
        targets = [boxes for boxes in targets if boxes is not None]
        # add sample index to targets
        for i, boxes in enumerate(targets):
            #print("-- i : ", i)
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
    
        # select new image size every tenth step
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        # resize images
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return imgs, targets



# sample of calling

if __name__ == "__main__":
    coco_dir = "/home/lampson/2T_disk/Data/COCO_CommonObjectsInContext/COCO_2014"
    dataset = "val2014"
    
    test_dataset = COCODatasetYolo(coco_dir, dataset)
    test_params = {
            "batch_size" : 4,
            "shuffle" : False,
            "drop_last" : False,
            "num_workers" : 0,
            "collate_fn" : test_dataset.collate_fn
            }
    test_generator = DataLoader(test_dataset, **test_params)
    
    for i, (img, targets) in enumerate(test_generator):
        #print("img : ", img.shape)
        print("targets : ", targets)
        a = 1
