import tensorflow as tf
import numpy as np
from models.basecolorizer import BaseColorizer
from config import colorization_consts
class LabColorizer(BaseColorizer):
    def __init__(self, path, weights_gen_class = None, neg_norm = True):
        if not weights_gen_class:
            super().__init__(path)
        else:
            size = (colorization_consts.IMAGE_HEIGHT['lab'], colorization_consts.IMAGE_HEIGHT['lab'], 1)
            self.model = weights_gen_class(size, path).model
        self.neg_norm = neg_norm

    def colorize(self, norm_gray_image):
        gray_image_tensor = tf.convert_to_tensor(norm_gray_image)

        gray_image_tensor = tf.expand_dims(gray_image_tensor, axis=-1)
        gray_image_tensor = tf.expand_dims(gray_image_tensor, axis=0)

        colorized_image_ab = self.model(gray_image_tensor, training=True)[0]
        norm_gray_image = np.reshape(norm_gray_image, (norm_gray_image.shape[0], norm_gray_image.shape[1], 1))
        norm_colorized_image = np.concatenate((norm_gray_image, colorized_image_ab), axis=2)
        
        if self.neg_norm:
            colorized_image_lab = (norm_colorized_image + [1.,1., 1.]) * [127.5, 127.5, 127.5]
        else:
            colorized_image_lab = (norm_colorized_image) * [255.0, 255.0, 255.0]


        return np.array(tf.cast(colorized_image_lab, tf.uint8))   



    