from config import colorization_consts
import tensorflow as tf
import numpy as np
from models.basecolorizer import BaseColorizer

class LabColorizer(BaseColorizer):
    def __init__(self):
        super().__init__(colorization_consts.LAB_MODEL_PATH)

    def colorize(self, norm_gray_image):
        # Convert numpy array to tensor
        gray_image_tensor = tf.convert_to_tensor(norm_gray_image)

        # # Add a batch dimension to the tensor
        gray_image_tensor = tf.expand_dims(gray_image_tensor, axis=-1)
        gray_image_tensor = tf.expand_dims(gray_image_tensor, axis=0)

        # Now you can input it to your model
        colorized_image_ab = self.model(gray_image_tensor, training=True)[0]
        norm_gray_image = np.reshape(norm_gray_image, (norm_gray_image.shape[0], norm_gray_image.shape[1], 1))
        norm_colorized_image = np.concatenate((norm_gray_image, colorized_image_ab), axis=2)
        # norm_colorized_image[:, :, 1:] = 0
        colorized_image_lab = (norm_colorized_image + [1.,1., 1.]) * [127.5, 127.5, 127.5]
        return np.array(tf.cast(colorized_image_lab, tf.uint8))   



    