
import tensorflow as tf
import numpy as np
from models.basecolorizer import BaseColorizer

class RGBColorizer(BaseColorizer):
    def __init__(self, model_path):
        super().__init__(model_path)

    def colorize(self, norm_gray_image):
        gray_image_tensor = tf.convert_to_tensor(norm_gray_image)

        gray_image_tensor = tf.expand_dims(gray_image_tensor, axis=0)

        colorized_image = self.model(gray_image_tensor, training=True)[0]
        colorized_image *= 255.0
        return np.array(tf.cast(colorized_image, tf.uint8))   



    