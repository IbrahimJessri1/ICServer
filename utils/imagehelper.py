import cv2
import numpy as np
from config import colorization_consts
from PIL import Image
from PIL import ImageEnhance
from utils.realesrgan import RealESRGANEnhancer
class ImageHelper:

    def resize_input(img, color):
        return cv2.resize(img, (colorization_consts.IMAGE_HEIGHT[color], colorization_consts.IMAGE_WIDTH[color]))
    

    def resize(img, size):
        return cv2.resize(img, size)
    

    def resize_image(image, size, enhancer : RealESRGANEnhancer = None ):
        # Convert BGR numpy array to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Clip the values to the range [0, 255] and convert data type to uint8
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
        
        # Convert RGB numpy array to a Pillow image
        pil_image = Image.fromarray(image_rgb)
        
        # Resize the image using Pillow
        if enhancer:
            pil_image = enhancer.enhance(pil_image)
        pil_image_resized = pil_image.resize(size, Image.BICUBIC)
    
        # enhancer = ImageEnhance.Sharpness(pil_image_resized)
        # pil_image_resized_sharp = enhancer.enhance(1.2)  # 2.0 is a sample factor; adjust as needed
        # enhancer = ImageEnhance.Contrast(pil_image_resized_sharp)
        # pil_image = enhancer.enhance(1.05)  # 1.1 is a sample factor; adjust as needed


        # Convert the resized Pillow image back to an RGB numpy array
        image_rgb_resized = np.array(pil_image_resized)
        
        # Convert the RGB numpy array back to BGR format
        image_bgr_resized = cv2.cvtColor(image_rgb_resized, cv2.COLOR_RGB2BGR)
        
        return np.array(image_bgr_resized).astype('float32')

    
    def rgb_normalize(img):
        return np.array(img.astype('float32') / 255.0)

    def bgr_to_gray(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def rgb_to_bgr(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def bgr_to_lab(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB) 
    
    def lab_to_bgr(img):
        return cv2.cvtColor(img, cv2.COLOR_LAB2BGR) 
    
    def read_image(image_data):
        np_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        return np.array(image).astype('float32')
    
    def image_to_output_file(image, content_type):
        if content_type == "image/png":
            extension = ".png"
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]  # No compression for PNG for maximum quality
            media_type = "image/png"
        else:
            extension = ".jpg"
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]  # Maximum quality for JPEG
            media_type = "image/jpeg"

        result, encoded_img = cv2.imencode(extension, image, encode_param)
        if not result:
            raise Exception(f"Error encoding the {extension} image")
        return encoded_img.tobytes(), media_type
    
    def denoise(img):
        img = np.array(img).astype('uint8')
        return cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 5, 11)

        
    