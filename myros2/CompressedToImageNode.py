import os
import datetime
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class CompressedToImageNode(Node):
    def __init__(self):
        super().__init__('compressed_to_image_node')
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
        self.publisher = self.create_publisher(Image, '/front/image', 10)
        self.get_logger().info(f"CompressedToImageNode 已启动，保存路径: {self.save_path}")

    def compressed_callback(self, msg: CompressedImage):
        try:
            msg_timestamp = msg.header.stamp
        # 将 ROS 时间戳转换为浮点数（秒）
            timestamp = msg_timestamp.sec + msg_timestamp.nanosec / 1e9
            # 解码压缩图像
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                self.get_logger().error("解码图片失败！")
                return

            # 保存图像到指定路径
            if self.save_path is not None:
                filename = os.path.join(
                    self.save_path, 
                    f"image_{self.image_count:04d}_{timestamp}.jpg"
                )
                success = cv2.imwrite(filename, cv_image)
                if success:
                    self.get_logger().info(f"图像已保存: {filename}")
                    self.image_count += 1  # 更新计数器
                else:
                    self.get_logger().error(f"保存失败: {filename}")

            # 转换并发布解码后的图像消息
            image_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            self.publisher.publish(image_msg)

        except Exception as e:
            self.get_logger().error(f"回调函数异常: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = CompressedToImageNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()