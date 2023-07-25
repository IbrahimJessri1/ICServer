
from config import colorization_consts
import tensorflow as tf
import numpy as np

class Colorizer:
    def __init__(self):
        self.model = tf.keras.models.load_model(colorization_consts.MODEL_PATH)

    def colorize(self, norm_gray_image):
        # Convert numpy array to tensor
        gray_image_tensor = tf.convert_to_tensor(norm_gray_image)

        # Add a batch dimension to the tensor
        gray_image_tensor = tf.expand_dims(gray_image_tensor, axis=0)

        # Now you can input it to your model
        colorized_image = self.model(gray_image_tensor, training=True)[0]
        colorized_image *= 255.0
        return np.array(tf.cast(colorized_image, tf.uint8))   



    