from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import argparse

import cv2
import torch
import numpy as np
from glob import glob

# os.chdir('/home/tong/project/pysot')
# sys.path.append('/home/tong/project/pysot/')
# sys.path.insert(0, '/home/tong/project/pysot') 

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

class Tracking(object):
    
    def __init__(self, config, snapshot):
        cfg.merge_from_file(config)
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        device = torch.device('cuda' if cfg.CUDA else 'cpu')

        # create model
        self.model = ModelBuilder()

        # load model
        self.model.load_state_dict(torch.load(snapshot,
            map_location=lambda storage, loc: storage.cpu()))
        self.model.eval().to(device)

        # build tracker
        self.tracker = build_tracker(self.model)
        self.center_pos = None
        self.size = None
          
    def init(self, frame, init_rect):
        
        print("initial rectangle selected as: ", init_rect)
        init_rect = list(map(int, init_rect.split(',')))
        self.tracker.init(frame, init_rect)
        
    def update(self, bbox):
        
        ## REMEMBER TO CALL UPDATE
        self.tracker.update(bbox)
    
    def get_roi(self, img, instance_size):
     
        return self.tracker.get_roi(img, instance_size)
        
    def track(self, frame, x_crop, scale_z, instance_size):
        # x_crop, scale_z = self.get_roi(frame)
        return self.tracker.track(frame, x_crop, scale_z, instance_size)
    
    # Following functions are used for template update
    def templateFeature(self, z):
        
        return self.model.templateFeature(z)
    
    def zf(self):
        
        return self.model.zf
    
    def updateTemplate(self, zf):
        
        model.zf = zf
        
        
    