

import os
from DEIM.tools.inference.onnx_inf import resize_with_aspect_ratio
from DEIM.tools.inference.onnx_inf import draw as onnx_draw
from DEIM.tools.inference.torch_inf_vis import draw as torch_draw
import onnxruntime as ort
from PIL import Image
import torch
import torchvision.transforms as T
from tqdm import tqdm
from utils.utils import mkdirs
import torch.nn as nn
import matplotlib.pyplot as plt

from DEIM.engine.core import YAMLConfig

config_path_dict = {'DEIM_S_onnx':'',
                    'DEIM_S':'DEIM/configs/deim_rtdetrv2/deim_r18vd_120e_coco.yml',
                    'RTDETR_S':'DEIM/configs/deim_rtdetrv2/rtdetrv2_r18vd_120e_coco.yml'}

model_path_dict = {'DEIM_S_onnx':'./checkpoints/deim_rtdetrv2_r18vd_coco_120e.onnx',
                   'DEIM_S':'checkpoints/deim_rtdetrv2_r18vd_coco_120e.pth',
                   'RTDETR_S':'checkpoints/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth'}
class DEIMProcess():
    def __init__(self,args):
        self.args = args
        self.model_name = args.model_name
        self.model_path = model_path_dict[self.model_name]
        self.device = torch.device(args.device)
        if 'onnx' in self.model_name:
            self.model = ort.InferenceSession(self.model_path)
        else:
            self.model = self.load_torch().to(self.device)
        

    def load_torch(self):
        config_path = config_path_dict[self.model_name]

        """Main function"""
        cfg = YAMLConfig(config_path, resume=self.model_path)

        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

        if self.model_path:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'ema' in checkpoint:
                state = checkpoint['ema']['module']
            else:
                state = checkpoint['model']
        else:
            raise AttributeError('Only support resume to load model.state_dict by now.')

        # Load train mode state and convert to deploy mode
        cfg.model.load_state_dict(state)
        return Model(cfg)

    def onnx_process_image(self, im_pil):
        # Resize image while preserving aspect ratio
        resized_im_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(im_pil, 640)
        orig_size = torch.tensor([[resized_im_pil.size[1], resized_im_pil.size[0]]])

        transforms = T.Compose([
            T.ToTensor(),
        ])
        im_data = transforms(resized_im_pil).unsqueeze(0)

        output = self.model.run(
            output_names=None,
            input_feed={'images': im_data.numpy(), "orig_target_sizes": orig_size.numpy()}
        )

        labels, boxes, scores = output

        result_images = onnx_draw(
            [im_pil], labels, boxes, scores,
            [ratio], [(pad_w, pad_h)]
        )
        return result_images[0]

    def torch_process_image(self,im_pil):
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(self.device)

        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms(im_pil).unsqueeze(0).to(self.device)

        output = self.model(im_data, orig_size)
        labels, boxes, scores = output[0]['labels'], output[0]['boxes'], output[0]['scores']
        vis_image = torch_draw(im_pil.copy(), labels, boxes, scores)
        return vis_image

    def process_image(self,im_pil):
        if 'onnx' in self.model_name:
            image = self.onnx_process_image(im_pil)
        else:
            image = self.torch_process_image(im_pil)
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


class Model(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.model = cfg.model.eval().cuda()
        self.postprocessor = cfg.postprocessor.eval().cuda()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs
