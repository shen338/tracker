import sys
import os
import cv2
import torch
import numpy as np
from glob import glob

sys.path.append('./tracking/')
print(sys.path)
from tracking.sot import Tracking
from reid import REID
from detection import Detection

tracker = Tracking(config='tracking/experiments/siamrpn_r50_l234_dwxcorr/config.yaml', 
                        snapshot='tracking/experiments/siamrpn_r50_l234_dwxcorr/model.pth')

detector = Detection(config="./detectron2/configs/COCO-InstanceSegmentation/small.yaml", 
                 model="detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl")

reid_module = REID(model='resnet18')