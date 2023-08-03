import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN
from config import colorization_consts


class RealESRGANEnhancer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = RealESRGAN(self.device, scale=4)
        self.model.load_weights(colorization_consts.SR_MODEL_PATH, download=False)

    def enhance(self, img):
        return self.model.predict(img)
