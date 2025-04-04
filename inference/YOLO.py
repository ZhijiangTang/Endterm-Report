
from ultralytics import YOLO
import cv2
import os
import sys
from utils.utils import mkdirs
from PIL import Image
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

label_map = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorbike', 5: 'aeroplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'trafficlight',
    11: 'firehydrant', 12: 'streetsign', 13: 'stopsign', 14: 'parkingmeter',
    15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse',
    20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra',
    25: 'giraffe', 26: 'hat', 27: 'backpack', 28: 'umbrella', 29: 'shoe',
    30: 'eyeglasses', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sportsball', 38: 'kite', 39: 'baseballbat',
    40: 'baseballglove', 41: 'skateboard', 42: 'surfboard', 43: 'tennisracket',
    44: 'bottle', 45: 'plate', 46: 'wineglass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hotdog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'sofa',
    64: 'pottedplant', 65: 'bed', 66: 'mirror', 67: 'diningtable', 68: 'window',
    69: 'desk', 70: 'toilet', 71: 'door', 72: 'tv', 73: 'laptop',
    74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cellphone', 78: 'microwave',
    79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 83: 'blender',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddybear',
    89: 'hairdrier', 90: 'toothbrush', 91: 'hairbrush'
}
label_map_reversed = {v: k for k, v in label_map.items()}

# 自动生成颜色（使用 matplotlib 的配色方案）
COLORS = plt.cm.tab20.colors  # 使用 20 种独特颜色，适合 CVPR 论文
COLOR_MAP = {label: tuple([int(c * 255) for c in COLORS[i % len(COLORS)]]) for i, label in enumerate(label_map)}


class YOLOProcess():
    def __init__(self,args):
        self.model_name = args.model_name
        self.model = YOLO("./checkpoints/yolov8n.pt")
        self.class_mapper = self.model.names

    def process_image(self,image):
        result = self.model(image,verbose=False)[0]
        boxes = result.boxes.xyxy  # 获取边界框 (x1, y1, x2, y2)
        labels = result.boxes.cls  # 获取类别索引
        scores = result.boxes.conf  # 获取置信度
        image = self.draw(image, labels, boxes, scores, thrh=0.01)
        return image

    def process_dir(self, dir):
        save_dir = os.path.join(*(os.path.split(dir)[:-1]),self.model_name)
        mkdirs(save_dir)
        img_names = [img for img in os.listdir(dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        if not img_names:
            print("未找到符合条件的图片文件！")
            return
        for img_name in tqdm(img_names):
            im_pil = Image.open(os.path.join(dir, img_name)).convert('RGB')
            image = self.process_image(im_pil)
            image.save(os.path.join(save_dir,img_name))


    def draw(self, image, labels, boxes, scores, thrh=0.01):
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()  # 可替换为更高质量的字体文件路径
        labels, boxes, scores = labels[scores > thrh], boxes[scores > thrh], scores[scores > thrh]

        for j, box in enumerate(boxes):
            category = labels[j].item()
            category_name = self.class_mapper[category]
            if category_name not in label_map_reversed:
                label_map_reversed[category_name] = len(category_name)+1
                label_map[len(category_name)+1] = category_name

            category = label_map_reversed[category_name]
            color = COLOR_MAP.get(category, (255, 255, 255))  # 默认白色
            box = list(map(int, box))

            # 画边框
            draw.rectangle(box, outline=color, width=3)
            
            # 添加标签和置信度
            text = f"{category_name} {scores[j].item():.2f}"
            text_bbox = draw.textbbox((0, 0), text, font=font)  # 获取文本边界框
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            # 添加文本背景
            text_background = [box[0], box[1] - text_height - 2, box[0] + text_width + 4, box[1]]
            draw.rectangle(text_background, fill=color)
            # 绘制文本
            draw.text((box[0] + 2, box[1] - text_height - 2), text, fill="black", font=font)

        return image


