from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

def get_subwindow(im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                             int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                          int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)

        return im_patch
    
def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        #raw_path = folder_path +"raw_640_480/"+"[0-9]*.jpeg"
        folder_path = video_name
        fields = folder_path.split("/")
        raw_path = folder_path+fields[-2]+"-" +"[0-9]*.jpeg"
        #print("raw_path = ", raw_path)

 

        raw_names = sorted(glob.glob(raw_path), key=lambda x:float(re.findall("([0-9]+?)\.jpeg",x)[0])) 
        images = raw_names
        #images = glob(os.path.join(video_name, '*.jp*'))
        #images = sorted(images)
        # images = sorted(images,
        #                 key=lambda x: int(x.split('/')[-1].split('.')[0]))
        print(images[0:10])
        for img in images:
            frame = cv2.imread(img)
            yield frame
#     else:
#         images = glob(os.path.join(video_name, '*.jp*'))
#         images = sorted(images)
#         # images = sorted(images,
#         #                 key=lambda x: int(x.split('/')[-1].split('.')[0]))
#         print(images[0:10])
#         for img in images:
#             frame = cv2.imread(img)
#             yield frame

            
            
import math

class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return math.sqrt(self.variance())
    
class Tracklet(object):
    
    def __init__(self, capacity):
        
        self.capacity = capacity
        self.size = 0
        self.frames = []
        self.ids = []

    def push_frame(self, frame, current_id):
        
        self.size += 1
        self.frames.append(frame)
        self.ids.append(current_id)
        
        if self.size > self.capacity: 
            self.size -= 1
            self.frames.pop(0)
            self.ids.pop(0)

    def get_features(self):
        return np.array(self.ids)

# Kalman filter for basic motion model 
class Kalman(object):
    
    def __init__(self, init_state, R, Q, dim_x=8, dim_z=4): 
        
        # State: dim_x = 8: [x, y, vel_x, vel_y, w, h, vel_w, vel_h]
        # Measurement: dim_z = 4: [x, y, w, h] 
        self.dim_x = dim_x  
        self.dim_z = dim_z 
        
        self.tracker = KalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z)
        
        f = np.array([[1, 1], [0, 1]])
        self.tracker.F = block_diag(f, f, f, f)
        # print(self.tracker.F)
        self.tracker.u = 0.
        self.tracker.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0]])
        
        self.tracker.R = R
        self.tracker.Q = Q
        # print(Q)

        self.tracker.x = np.array([[init_state[0], 0, init_state[1], 0, init_state[2], 0, init_state[3], 0]]).T
        p = np.zeros((8, 8), int)
        np.fill_diagonal(p, np.array([1, 10, 1, 10, 1, 10, 1, 10]))
        # print(p)
        self.tracker.P = p
        
    def predict(self):
        
        self.tracker.predict()
        return self.tracker.x
    
    def update(self, new_state):
        
        self.tracker.update(new_state)

# Dense optical flow on examplar object for motion prediction 
class OpticalFlow(object): 
    
    def __init__(self, init_frame, init_rect):
        
        self.prev = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
        self.prev_rect = init_rect
        
    # predict x y coordinate change
    def predict(self, frame): 
        
        example = frame[self.prev_rect[1]:self.prev_rect[3], self.prev_rect[0]:self.prev_rect[2], :]
        example = cv2.cvtColor(example, cv2.COLOR_BGR2GRAY)
        # print(example.shape, self.prev.shape)
        flow = cv2.calcOpticalFlowFarneback(self.prev, example, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        
        scale = np.mean(mag) 
        x = - np.sin(np.mean(ang) + np.pi/2)*scale
        y = np.cos(np.mean(ang) + np.pi/2)*scale
        
        return x, y
    
    def update(self, target, rect):
        # REMEMBER to call update 
        
        self.prev = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        self.prev_rect = rect
        
        
def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1

def drawpoly(img,pts,color,thickness=1,style='dotted',):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness,style)

def drawrect(img,pt1,pt2,color,thickness=1,style='dashed'):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])] 
    drawpoly(img,pts,color,thickness,style)

SMOOTH = 1e-6
def IOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = (interArea + SMOOTH) / float(boxAArea + boxBArea - interArea + SMOOTH)
    # return the intersection over union value
    return iou
    
    