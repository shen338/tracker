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

import torchreid


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
    
    # Build reid module
    reid_datamanager = torchreid.data.ImageDataManager(
        root='/home/tong/deep-person-reid/car',
        sources='market1501',
        targets='market1501',
        height=128,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop'])
    
    reid_model = torchreid.models.build_model(
        name='resnet18',
        num_classes=reid_datamanager.num_train_pids,
        loss='softmax',
        pretrained=True
    )

    reid_model = reid_model.cuda()

    reid_optimizer = torchreid.optim.build_optimizer(
        reid_model,
        optim='adam',
        lr=0.0003
    )

    reid_scheduler = torchreid.optim.build_lr_scheduler(
        reid_optimizer,
        lr_scheduler='single_step',
        stepsize=20
    )

    reid_engine = torchreid.engine.ImageSoftmaxEngine(
        reid_datamanager,
        reid_model,
        optimizer=reid_optimizer,
        scheduler=reid_scheduler,
        label_smooth=True
    )
    reids = []
    for frame in get_frames(args.video_name):
        count += 1
        if count > 500:
            break
        if first_frame:
            frame_size = frame.shape
            print(frame_size)
            out = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*'DIVX'), 30, (frame_size[1], frame_size[0]))
            tracker.init(frame, init_rect)
            first_frame = False
        
        else:
            outputs = tracker.track(frame)
            bbox = list(map(int, outputs['bbox']))
            # print(bbox)
            target = frame[max(0, bbox[1]):(bbox[1]+bbox[3]), max(0, bbox[0]):(bbox[0]+bbox[2]), :]
            
            target = cv2.resize(target, (128, 128))
            
            cv2.imwrite("save_person/" + '{0:03d}'.format(count)+ ".jpg", target)
            target = torch.from_numpy(target).float().to(device).permute(2, 0, 1)
            target = target[None]
            # print(target.shape)
            f = reid_engine._extract_features(target)
            reids.append(f.squeeze().cpu().detach().numpy())
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)
            # cv2.imwrite("test.jpg", frame)
        out.write(frame)
         
    out.release()
    np.savetxt("save_person/reid.txt", np.array(reids))


if __name__ == '__main__':
    main()
