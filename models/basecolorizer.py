from config import colorization_consts
import tensorflow as tf

class BaseColorizer:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(path)

    def colorize(self, norm_gray_image):
        pass