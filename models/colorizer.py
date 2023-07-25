
from config import colorization_consts
import tensorflow as tf

class Colorizer:
    def __init__(self):
        self.model = tf.keras.models.load_model(colorization_consts.MODEL_PATH)

    