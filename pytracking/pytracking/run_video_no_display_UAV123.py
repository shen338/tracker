import os
import sys
import argparse
import glob 

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Tracker


def run_video(tracker_name, tracker_param, videofile, output_video, optional_box=None, debug=None, save_results=False):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    tracker = Tracker(tracker_name, tracker_param)
    tracker.run_video_no_display(videofilepath=videofile, output_video=output_video, optional_box=optional_box, debug=debug, save_results=save_results)

def main():
    
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('videofile', type=str, help='path to a video file.')
    parser.add_argument('--output_video', type=str, default=None, help='output video name')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
    parser.set_defaults(save_results=True)

    args = parser.parse_args()
    folders = []
    visdrone_sot_dir = '../../dataset/dataset/UAV123/*/'
    folders += glob.glob(visdrone_sot_dir)
    video_dir = "../../video/"
    print(folders)
    # print(folders[0].split('/')[-2])
    
    for folder in folders: 
        basename = folder.split('/')[-2]
        filename = glob.glob(os.path.join(video_dir, basename + "*.txt"))
        filename = sorted(filename)[0]
        with open(filename) as f:
            content = f.readlines()
            
        optional_box = list(map(int, content[0].split(',')))
        # optional_box = " ".join(bbox)
        print(basename)
        print(optional_box)
        
        run_video(args.tracker_name, args.tracker_param, video_dir + basename + ".avi", basename + ".avi", optional_box, args.debug, args.save_results)


if __name__ == '__main__':
    main()
