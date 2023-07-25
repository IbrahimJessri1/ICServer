import cv2
import numpy as np

class ImageHelper:

    def resize(img, size):
        return cv2.resize(img, size)

    def rgb_normalize(img):
        return np.array(img.astype('float32') / 255.0)

    def rgb_to_gray(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    def read_image(image_data):
        np_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        return np.array(image).astype('float32')
    
    def image_to_output_file(image, content_type):
        if content_type == "image/png":
            extension = ".png"
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
            media_type = "image/png"
        else:
            extension = ".jpg"
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            media_type = "image/jpeg"

        result, encoded_img = cv2.imencode(extension, image, encode_param)
        if not result:
            raise Exception(f"Error encoding the {extension} image")
        return encoded_img.tobytes(), media_type
        
    