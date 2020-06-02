# created by lampson.song @ 2020-3-16
# load trained yolov3-spp and run an image

import torch
from net.yolov3_spp import  YOLOv3_SPP
import os

from utils.utils import non_max_suppression, scale_coords, clip_coords
from dataset.coco_dataset_yolo import letterbox

import cv2
import numpy as np
import sys

image_demo = False
video_demo = False
if (len(sys.argv) < 2):
    print("Usage: python demo.py [image|video] test.jpg|test.mp4")
else:
    if(sys.argv[1] == 'image'):
        image_demo = True
    if(sys.argv[1] == 'video'):
        video_demo = True

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']


def process_img(img):
    input_shape = (416, 416)
    #input_shape = (608, 608)
    #input_shape = (960, 960)
    
    resized_img, ratio, _ = letterbox(img, input_shape, auto=False, scaleFill=False)
    
    resized = resized_img[:, :, ::-1].transpose(2,0,1) # BGR to RGB
    resized = np.ascontiguousarray(resized)
    
    input_data = torch.from_numpy(resized).to(device)
    input_data = input_data.float()
    input_data /= 255.
    
    if input_data.ndimension() == 3:
        input_data.unsqueeze_(0)
    
    input_data.to(device)
    
    out = yolov3spp(input_data)
    
    out = non_max_suppression(out, conf_thres=0.2)
    
    for i, det in enumerate(out):
        if det is not None and len(det):
            det[:, :4] = scale_coords(input_shape, det[:, :4], img.shape).round()
            
            for x1, y1, x2, y2, conf, cls in det:
                cv2.rectangle( img, (x1, y1), (x2, y2), (0,0,255), 2)
                cls_num = int(cls.cpu().detach().numpy())
                cv2.putText(img, obj_list[cls_num], (x1,y1), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255), 0)
    
    cv2.imshow("i", img)
    k = cv2.waitKey(0)






device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

yolov3spp = YOLOv3_SPP(num_classes=80)
yolov3spp.to(device).eval()

trained_model_path = "./yolov3-best.pt"

if os.path.exists(trained_model_path):
    trained_model = torch.load(trained_model_path, map_location=device)['model']
    yolov3spp.load_state_dict(trained_model)

if image_demo:
    if sys.argv[2] is None:
        f_path = "./test.jpg"
    else:
        f_path = sys.argv[2]
    img = cv2.imread(f_path)

    process_img(img)
if video_demo:
    if sys.argv[2] is None:
        f_path = "./test.mp4"
    else:
        f_path = sys.argv[2]
    cap = cv2.VideoCapture(f_path)

    while True:
        _, img = cap.read()
        process_img(img)


