import os
import datetime
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image as ROSImage
from PIL import Image as PILImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
sys.path.append('../')
os.chdir('/root/ros/code/Endterm-Report')

import io

class DetectionImageNode(Node):
    def __init__(self,args):
        super().__init__('detection_image_node')
        self.model_name = args.model_name
        if 'YOLO' in args.model_name:
            from inference.YOLO import YOLOProcess
            self.process = YOLOProcess(args)
        elif 'image' in args.model_name:
            from inference.IMAGE import IMAGEProcess
            self.process = IMAGEProcess()
        else:
            from inference.DEIM import DEIMProcess
            self.process = DEIMProcess(args)
        self.bridge = CvBridge()

        # 1. 读取保存路径参数（默认保存到当前目录的 "saved_images" 文件夹）
        self.save_path = self.declare_parameter('image_save_path', None).value
        if self.save_path is not None:
            self.image_count = 0
            os.makedirs(self.save_path, exist_ok=True)  # 确保目录存在

        # 2. 订阅压缩图像话题
        self.subscription = self.create_subscription(
            CompressedImage,
            '/front/compressed',
            self.compressed_callback,
            10)
        
        # 3. 发布解码后的图像到 /front/image
        self.publisher = self.create_publisher(ROSImage, f'/front/{args.model_name}', 10)
        self.get_logger().info(f"CompressedToImageNode 已启动")

    def compressed_callback(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            pil_image = PILImage.open(io.BytesIO(np_arr)).convert('RGB')  # 使用 io.BytesIO 解压数据为 PIL Image
            vis_image = self.process.process_image(pil_image)
            # 如果你需要对图像进行广播或其他处理，可以在此处进行
            # 如果你要广播为 RGB 图像
            np_image = np.array(vis_image)

            # 将 NumPy 数组转换为 ROS 图像消息（使用 CvBridge 转换）
            image_msg = self.bridge.cv2_to_imgmsg(np_image, encoding="bgr8")
            self.publisher.publish(image_msg)

        except Exception as e:
            self.get_logger().error(f"回调函数异常: {str(e)}")

def detection_main(args=None):
    rclpy.init(args=None)
    node = DetectionImageNode(args)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
