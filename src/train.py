# created by lampson.song @ 2020-3-23
# training scripts of YOLOv3

import argparse
from dataset.coco_dataset_yolo import COCODatasetYolo

def train_yolo(opt):

    gpu_ids = [int(x) for x in opt.gpus.split(',')]
    print("-- gpu_ids : ",gpu_ids)

    ## train dataloader
    #train_loader = COCODatasetYolo(coco_dir=opt.coco_dir, set_name='train2017', img_size=opt.img_size, phase='Train')
    #test_loader = COCODatasetYolo(coco_dir=opt.coco_dir, set_name='val2017', img_size=opt.img_size, phase='Test')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_dir', type=str, default='/home/lampson/2T_disk/Data/COCO_CommonObjectsInContext/COCO_2017')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=416)
    parser.add_argument('--gpus', type=str, default='0,1,2')
    opt = parser.parse_args()

    train_yolo(opt)

    print("done")
