import sys
import os
import cv2
import torch
import numpy as np
from glob import glob

sys.path.append('/home/tong/project/tracking/')
print(sys.path)
from tracking.sot import Tracking
from reid import REID
from detection import Detection
from utils import get_frames, RunningStats, Tracklet, OpticalFlow, Kalman
from PIL import Image

from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

# TRACKER PARAMETERS
EXAMPLAR_SIZE = 127
INSTANCE_SIZE = 255
LOST_INSTANCE_SIZE = 512 # original value is 831
WINDOW_INFLUENCE = 0.35

# KALMAN PARAMETERS
KALMAN_RATIO = 0.7
MEASUREMENT_STD = 3
PROCESS_STD = 1
TRACKLET_SIZE = 30

# REID PARAMETERS
REID_INSTANCE_SIZE = 128

# DETECTION PARAMETERS 
DETECTION_SIZE = 256


tracker = Tracking(config='tracking/experiments/siamrpn_r50_l234_dwxcorr/config.yaml', 
                        snapshot='tracking/experiments/siamrpn_r50_l234_dwxcorr/model.pth')

detector = Detection(config="./detectron2/configs/COCO-InstanceSegmentation/small.yaml", 
                 model="detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl")

reid_module = REID(model='resnet18')

tracklet = Tracklet(TRACKLET_SIZE)

running_stats = RunningStats()

def reid_rescore(reid_module, frame, template_features, bboxes, scores):
    
    #  rescore detection and tracking results with REID module and sort results. 
    batch = []
    for bbox in bboxes: 
        
        target = frame[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        # print(target.shape)
        target = cv2.resize(target, (128, 128))
        batch.append(target)
    batch = np.array(batch).astype(np.float32)
    # BGR to RGB
    batch = batch[:, :, :, ::-1]
    # BHWC to BCHW
    batch = batch.transpose(0, 3, 1, 2)
    # To image of range 0-1
    batch = np.divide(batch, 255)
    # Normalize with ImageNet norm_mean=[0.485, 0.456, 0.406], and norm_std=[0.229, 0.224, 0.225],
    norm_mean = np.array([0.485, 0.456, 0.406])[None, :, None, None]
    norm_std = np.array([0.229, 0.224, 0.225])[None, :, None, None]
    batch = (batch - norm_mean)/norm_std
    # To Tensor
    # batch = batch.astype(float)
    batch = torch.from_numpy(batch).float().cuda()
    # batch = np.array(batch)
    # batch = torch.from_numpy(batch)
    # batch = Image.fromarray(batch)
    target_features = reid_module.extract_feature(batch)
    target_features = target_features.cpu().detach().numpy()
    
    similarity = np.dot(template_features, target_features.transpose())
    # print(similarity.shape)
    similarity = np.mean(similarity, axis=0)
    # print(similarity.shape)
    
    scores *= np.squeeze(similarity)
    return scores, target_features

video_name = '../tracking_dataset/car/car-14/img/'
init_string = '614,389,103,99'
output_video = 'video.avi'
out = None
init_rect = list(map(int, init_string.split(',')))
count = 0
first_frame = True

import time 
for frame in get_frames(video_name):
    
    count += 1

    if first_frame:
        
        frame_size = frame.shape
        init_rect = list(map(int, init_string.split(',')))
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'DIVX'), 30, (frame_size[1], frame_size[0]))
        tracker.init(frame, init_string)
        first_frame = False
        target = frame[init_rect[1]:init_rect[3]+init_rect[1], init_rect[0]:init_rect[2]+init_rect[0], :]
        # cv2.imwrite("target.jpg", target)
        # initialize Optical Flow
        optical_flow = OpticalFlow(target, [init_rect[0], init_rect[1], init_rect[2]+init_rect[0], init_rect[3]+init_rect[1]])
       
        # Put first frame into key frame pool 
        target = cv2.resize(target, (128, 128))
        
        batch = np.array(target).astype(np.float32)[None, :, :, :]
        # BGR to RGB
        batch = batch[:, :, :, ::-1]
        # BHWC to BCHW
        batch = batch.transpose(0, 3, 1, 2)
        # To image of range 0-1
        batch = np.divide(batch, 255)
        # Normalize with ImageNet norm_mean=[0.485, 0.456, 0.406], and norm_std=[0.229, 0.224, 0.225],
        norm_mean = np.array([0.485, 0.456, 0.406])[None, :, None, None]
        norm_std = np.array([0.229, 0.224, 0.225])[None, :, None, None]
        batch = (batch - norm_mean)/norm_std
        # To Tensor
        # batch = batch.astype(float)
        batch = torch.from_numpy(batch).float().cuda()
        # batch = np.array(batch)
        # batch = torch.from_numpy(batch)
        # batch = Image.fromarray(batch)
        target_features = reid_module.extract_feature(batch)
        target_features = target_features.cpu().detach().numpy()
        # print(target_features.shape)
        
        tracklet.push_frame(target, np.squeeze(target_features))
        
        # initialize 1st order Kalman filter with state [center_x, center_y, w, h] 
        dt = 1
        R = np.eye(4) * MEASUREMENT_STD**2
        q = Q_discrete_white_noise(dim=2, dt=1, var=PROCESS_STD**2)
        Q = block_diag(q, q, q, q)
        kalman_filter = Kalman([init_rect[1] + init_rect[3]/2, init_rect[0] + init_rect[2]/2, init_rect[3], init_rect[2]], R, Q)  

    else:
        
        tt = time.time()
        
        center_pos_correction = np.array([0.0, 0.0])
        # get optical flow result
        opt_flow_x, opt_flow_y = optical_flow.predict(frame)
        # print("opt: ", opt_flow_x, opt_flow_y)
        center_pos_correction += (1 - KALMAN_RATIO)*np.array([opt_flow_x, opt_flow_y])
        
        # only count Kalman after a few frames 
        # only use center position for now 
        kalman_pred = None
        if count > 5: 
            kalman_pred = np.squeeze(kalman_filter.predict())
            center_pos = tracker.tracker.center_pos
            # print("kalman: ", kalman_pred)
            center_pos_correction += KALMAN_RATIO*np.array([kalman_pred[0] - center_pos[0], kalman_pred[2] - center_pos[1]])
      
        # apply center position correction with Kalman and opt flow 
        # print(center_pos_correction)
        # center_pos_correction = np.cast(center_pos_correction, np.int)
        tracker.tracker.center_pos += center_pos_correction
        
        x_crop, scale_z = tracker.get_roi(frame)
        # print(scale_z)
        detection_result = detector.detect(np.squeeze(x_crop))
        # cv2.imwrite("test.jpg", np.squeeze(x_crop).transpose((1,2,0)))
        dboxes = detection_result["instances"].pred_boxes.tensor.cpu().detach().numpy()
        dscores = detection_result["instances"].scores.cpu().detach().numpy()
        center_pos = tracker.tracker.center_pos
            
        # print(tracker.tracker.center_pos, tracker.tracker.size)
        # print(dboxes, dscores)
        # print(x_crop.shape)
        all_outputs = tracker.track(frame, x_crop, scale_z)
        # cv2.imwrite("test.jpg", frame)
        tboxes, tscores = all_outputs['bbox'], all_outputs['best_score']
        # print(tboxes, tscores)
        
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

        
        for idx, dbox in enumerate(dboxes):
            for tbox in tboxes: 
                if IOU(dbox, tbox) > 0.8: 
                    dscores[idx] = 1   
        
        # Windows penalty for detection results
        
        input_size = 255
        hanning = np.hanning(input_size)
        window = np.outer(hanning, hanning)

        idx = 0
        for idx, dbox in enumerate(dboxes):
            x = int((dbox[0] + dbox[2])/2)
            y = int((dbox[1] + dbox[3])/2)
            # print(x, y, idx)
            dscores[idx] = dscores[idx] * (1 - WINDOW_INFLUENCE) + window[x, y]*WINDOW_INFLUENCE
        # print(dscores, tscores)
        # print(dscores, dboxes)
        # print(tscores, tboxes)
        
        # Detection and tracking result merge
        for dbox in dboxes:
            dbox[0] /= scale_z
            dbox[1] /= scale_z
            dbox[2] /= scale_z
            dbox[3] /= scale_z
            dbox[0] += center_pos[0] - 255/2/scale_z
            dbox[1] += center_pos[1] - 255/2/scale_z
            dbox[2] += center_pos[0] - 255/2/scale_z
            dbox[3] += center_pos[1] - 255/2/scale_z
              
        for tbox in tboxes: 
            tbox[2] += tbox[0]
            tbox[3] += tbox[1]
            
        overall_box = np.concatenate((dboxes, tboxes), axis=0).astype(int)
        overall_box[overall_box < 0] = 0
        overall_score = np.concatenate((dscores, tscores), axis=0)
        
        # Get key frame RE-ID from tracklet
        template_features = tracklet.get_features()
        # print(template_features.shape)
        print(frame.shape, overall_box)
        after_reid_score, reid_features = reid_rescore(reid_module, frame, template_features, overall_box, overall_score)
        
        best_idx = np.argmax(after_reid_score)
        best_bbox = overall_box[best_idx]
        best_score = after_reid_score[best_idx]
        running_stats.push(best_score)
        
        # Score better than one sigma, treat as key frame 
        if best_score >= running_stats.mean() + running_stats.standard_deviation():
            tracklet.push_frame(frame, np.squeeze(reid_features[best_idx]))
            
        # update tracker size and position with current best box
        tracker.update(best_bbox)
        # print(frame.shape, best_bbox)
        optical_flow.update(frame[best_bbox[1]:best_bbox[3], best_bbox[0]:best_bbox[2], :], best_bbox)
        cv2.imwrite("result.jpg", frame[best_bbox[1]:best_bbox[3], best_bbox[0]:best_bbox[2], :])
        
        kalman_filter.update([(best_bbox[0]+best_bbox[2])/2, (best_bbox[1]+best_bbox[3])/2, best_bbox[2] - best_bbox[0], best_bbox[3] - best_bbox[1]])
        
        # print(time.time() - tt)
        for outputs in tboxes:

            bbox = list(map(int, outputs))
            # print(bbox)
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
            
        for outputs in dboxes:

            bbox = list(map(int, outputs))
            # print(bbox)
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (255, 0, 0), 2)
            
        cv2.rectangle(frame, (best_bbox[0], best_bbox[1]),
                          (best_bbox[2], best_bbox[3]),
                          (0, 0, 255), 2)
        
    out.write(frame)

out.release()