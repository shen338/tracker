import subprocess
import os
import sys
import argparse
import glob 

folders = []
visdrone_sot_dir = '../dataset/dataset/VisDrone-SOT/*/'
folders += glob.glob(visdrone_sot_dir)
visdrone_ch_dir = '../dataset/dataset/VisDrone-Challenge/*/'
folders += glob.glob(visdrone_ch_dir)
visdrone_uavs_dir = '../dataset/dataset/UAV-benchmark-S/*/'
folders += glob.glob(visdrone_uavs_dir)
video_dir = "../video/"
print(folders)
# print(folders[0].split('/')[-2])

for folder in folders: 
    basename = folder.split('/')[-2]
    filename = glob.glob(os.path.join(video_dir, basename + "*.txt"))[0]
    with open(filename) as f:
        content = f.readlines()

    optional_box = list(content[0].split(','))
    optional_box = " ".join(optional_box)
    # print(folder)
    print(basename)
    print(optional_box)
    
    video = video_dir + basename + ".avi"

    command = "CUDA_VISIBLE_DEVICES=1 python3 ./demo/main/video/sot_video.py --config 'experiments/siamfcpp/test/trackingnet/siamfcpp_googlenet-trackingnet-fulldata.yaml' --dump-only --device cuda --video " + video + " --output " +  basename + ".avi" + " --init-bbox " + optional_box
    
    print(command)
    os.system(command)