import subprocess
import os
import sys
import argparse
import glob 


lasot_dir = '../dataset/dataset/LaSOT/*/'
folders = glob.glob(lasot_dir)
video_dir = "../video/"
print(folders)
# print(folders[0].split('/')[-2])

for folder in folders: 
    basename = folder.split('/')[-2]
    filename = os.path.join(folder, "groundtruth.txt")
    with open(filename) as f:
        content = f.readlines()

    optional_box = list(content[0].split(','))
    optional_box = " ".join(optional_box)
    # print(folder)
    print(basename)
    print(optional_box)
    
    video = video_dir + basename + ".avi"

    command = "python3 ./demo/main/video/sot_video.py --config 'experiments/siamfcpp/test/trackingnet/siamfcpp_googlenet-trackingnet-fulldata.yaml' --dump-only --device cuda --video " + video + " --output " +  basename + ".avi" + " --init-bbox " + optional_box
    
    print(command)
    os.system(command)