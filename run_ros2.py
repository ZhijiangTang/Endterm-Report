from myros2.RealTimeDetection import detection_main


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,default='DEIM_S', required=False, help='Path to the model file.')
    parser.add_argument('--device', type=str,default='cuda:0', required=False, help='Dir to the input image')
    args = parser.parse_args()
    detection_main(args)