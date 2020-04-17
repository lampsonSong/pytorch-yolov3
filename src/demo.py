# created by lampson.song @ 2020-3-16
# load trained yolov3-spp and run an image

import torch
from net.yolov3_spp import  YOLOv3_SPP
import os

from utils.utils import non_max_suppression, scale_coords, clip_coords
from dataset.coco_dataset_yolo import letterbox

import cv2
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

yolov3spp = YOLOv3_SPP(num_classes=80)
yolov3spp.to(device).eval()

trained_model_path = "/home/lampson/workspace-ln/objectDetection/YOLO/ultralytics-yolov3/weights/yolov3_model/416/yolov3-spp-ultralytics.pt"

if os.path.exists(trained_model_path):
    trained_model = torch.load(trained_model_path, map_location=device)['model']
    yolov3spp.load_state_dict(trained_model)


img = cv2.imread("./poseTesting.jpg")
input_shape = (416, 416)

resized_img, ratio, _ = letterbox(img, input_shape, auto=False, scaleFill=False)
cv2.imshow("resized", resized_img)

resized = resized_img[:, :, ::-1].transpose(2,0,1) # BGR to RGB
resized = np.ascontiguousarray(resized)

input_data = torch.from_numpy(resized).to(device)
input_data = input_data.float()
input_data /= 255.

if input_data.ndimension() == 3:
    input_data.unsqueeze_(0)

input_data.to(device)

out = yolov3spp(input_data)

out = non_max_suppression(out, conf_thres=0.5)

for i, det in enumerate(out):
    if det is not None and len(det):
        det[:, :4] = scale_coords(input_shape, det[:, :4], img.shape).round()
        
        for x1, y1, x2, y2, conf, cls in det:
            cv2.rectangle( img, (x1, y1), (x2, y2), (0,0,255), 2)

cv2.imshow("i", img)
cv2.waitKey(0)
