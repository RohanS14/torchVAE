# utils/transforms.py

import torch
import random
from torchvision import transforms

class BlackoutWithMask:
    def __init__(self, p=1.0, scale=(0.1, 0.2), ratio=(0.8, 1.2), value=0.0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img):
        img = transforms.ToTensor()(img)
        mask = torch.ones_like(img)

        if random.random() < self.p:
            _, H, W = img.shape
            area = H * W
            erase_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            h_erase = int((erase_area * aspect_ratio) ** 0.5)
            w_erase = int((erase_area / aspect_ratio) ** 0.5)
            top = random.randint(0, H - h_erase)
            left = random.randint(0, W - w_erase)

            img[:, top:top + h_erase, left:left + w_erase] = self.value
            mask[:, top:top + h_erase, left:left + w_erase] = 0.0

        return img, mask
