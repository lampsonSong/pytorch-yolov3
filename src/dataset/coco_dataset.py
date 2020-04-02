# created by lampson.song @ 2020-3-3
# read coco for pytorch

import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
import cv2

from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import random

class COCODataset(Dataset):
    def __init__(self, coco_dir, set_name='val2014', img_size=416):
        self.coco_dir = coco_dir
        self.set_name = set_name
        self.img_size = img_size
        self.coco = COCO(os.path.join(self.coco_dir, 'annotations', 'instances_'+self.set_name+'.json'))
        self.image_ids = self.coco.getImgIds()
        self.batch_count = 0

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])
    
        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)
    
        self.num_classes = len(self.classes)
        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def __getitem__(self, index):
        img = self.load_image(index)
        targets = self.load_box_annotation(index)

        return img, targets

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, index):
        img_info = self.coco.loadImgs(self.image_ids[index])[0]
        img_path = os.path.join(self.coco_dir, 'images', self.set_name, img_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (int(self.img_size), int(self.img_size)))
        return torch.tensor(img)

    def load_box_annotation(self, index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[index], iscrowd=False)
        annotations = np.zeros((0, 6))
 
        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return torch.FloatTensor(annotations)
 
        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations): 
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
               continue

            annotation = np.zeros((1, 6))
            annotation[0, 2:6] = a['bbox']
            annotation[0, 1] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        #print("-- annotations.shape : ", annotations.shape)
        # return (type, lx,ly,w,h) , shape is [box_num, 5]
        return torch.FloatTensor(annotations)

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        #print("== targets : ", targets)
        
        # remove empty placeholder targets, from tuple to list
        targets = [boxes for boxes in targets if boxes is not None]
        # add sample index to targets
        for i, boxes in enumerate(targets):
            #print("-- i : ", i)
            boxes[:, 0] = i
        #print(targets)
        #print("-----------------------------------------------")
        targets = torch.cat(targets, 0)
    
        # resize images
        imgs = torch.stack([img for img in imgs])
        self.batch_count += 1
        return imgs, targets


# sample of calling

if __name__ == "__main__":
    coco_dir = "/home/lampson/2T_disk/Data/COCO_CommonObjectsInContext/COCO_2014"
    dataset = "val2014"
    
    test_dataset = COCODataset(coco_dir, dataset)
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
