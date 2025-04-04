from inference.DEIM import DEIMProcess
from inference.YOLO import YOLOProcess

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,default='DEIM_S', required=False, help='Path to the model file.')
    parser.add_argument('--dir', type=str,default='./data/image', required=False, help='Dir to the input image')
    parser.add_argument('--device', type=str,default='cuda:0', required=False, help='Dir to the input image')
    args = parser.parse_args()
    if 'YOLO' in args.model_name:
        process = YOLOProcess(args)
    else:
        process = DEIMProcess(args)
    process.process_dir(dir=args.dir)