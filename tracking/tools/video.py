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

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
parser.add_argument('--init_rect', default='', type=str,
                    help='initial rectangle of target image. Format: [x1, y1, w, h]')
parser.add_argument('--output_video', default='', type=str,
                    help='video output')
args = parser.parse_args()


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
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images)
        # images = sorted(images,
        #                 key=lambda x: int(x.split('/')[-1].split('.')[0]))
        print(images[0:10])
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    # cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    out = None
    init_string = args.init_rect
    init_rect = list(map(int, init_string.split(',')))
    print("initial rectangle selected as: ", init_rect)
    print("output video is: ", args.output_video)
    count = 0
    for frame in get_frames(args.video_name):
        count += 1
#         if count < 100:
#             continue
        if first_frame:
            frame_size = frame.shape
            print(frame_size)
            out = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*'DIVX'), 30, (frame_size[1], frame_size[0]))
            tracker.init(frame, init_rect)
            first_frame = False
        
        else:
            all_outputs = tracker.track(frame)
            for outputs in all_outputs['bbox']:
#             if 'polygon' in outputs:
#                 polygon = np.array(outputs['polygon']).astype(np.int32)
#                 cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
#                               True, (0, 255, 0), 3)
#                 mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
#                 mask = mask.astype(np.uint8)
#                 mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
#                 frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            
                bbox = list(map(int, outputs))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 2)
            cv2.imwrite("test.jpg", frame)
        out.write(frame)
         
    out.release()


if __name__ == '__main__':
    main()
