import os
from utils.utils import images_to_video

if __name__ == '__main__':
    os.chdir('/root/ros/code/Endterm-Report')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str,default='DEIM_S_onnx', required=False, help='Path to the model file.')
    args = parser.parse_args()

    images_to_video(args.image_folder)