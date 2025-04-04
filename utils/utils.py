


import cv2
import os
from tqdm import tqdm


def images_to_video(image_folder: str, fps: int = 30):
    """
    将指定文件夹中的图片合成视频。
    
    :param image_folder: 存放图片的文件夹路径。
    :param video_path: 输出视频的路径（支持 .mp4 格式）。
    :param fps: 视频帧率，默认为 30。
    """
    # 获取所有图片并按名称排序（确保名称有序，如 frame_001.jpg）
    video_path = image_folder+'.mp4'
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])
    
    if not images:
        print("未找到符合条件的图片文件！")
        return
    
    # 读取第一张图片，获取视频宽高
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape
    
    # 定义视频编码格式并创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 或者 "XVID"
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # 逐帧写入图片
    for image in tqdm(images):
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"跳过无法读取的图片: {img_path}")
            continue
        video.write(frame)
    
    # 释放资源
    video.release()
    print("视频已生成:", video_path)

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

