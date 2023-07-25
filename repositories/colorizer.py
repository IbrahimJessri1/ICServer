
from models.basecolorizer import BaseColorizer
from utils.imagehelper import ImageHelper
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
import numpy as np

def colorize(colorizer_model : BaseColorizer, image_data, content_type, rgb = True):
    if rgb:
        colorized_image = colorize_rgb(colorizer_model, image_data)
    else:
        colorized_image = colorize_lab(colorizer_model, image_data)
    try:
        output_image_bytes, media_type = ImageHelper.image_to_output_file(colorized_image, content_type)
    except:
        raise HTTPException(status_code=400, detail="Something went wrong..")
    return StreamingResponse(BytesIO(output_image_bytes), media_type=media_type)

def colorize_rgb(colorizer_model, image_data):
    image = ImageHelper.read_image(image_data)
    input_image = ImageHelper.bgr_to_gray(image)
    size = input_image.shape
    input_image = ImageHelper.resize_input(input_image, 'rgb')
    norm_input_image = ImageHelper.rgb_normalize(input_image)
    colorized_image = colorizer_model.colorize(norm_input_image)
    colorized_image = ImageHelper.rgb_to_bgr(colorized_image)
    return ImageHelper.resize(colorized_image, (size[1], size[0]))

def colorize_lab(colorizer_model, image_data):
    image = ImageHelper.read_image(image_data) / 255.0
    input_image = ImageHelper.bgr_to_lab(image)
    size = input_image.shape
    input_image = input_image[:, :, 0]
    input_image = ImageHelper.resize_input(input_image, 'lab')
    input_image *= 2.0 / 100.0 
    norm_input_image = input_image - 1.0
    colorized_image = colorizer_model.colorize(norm_input_image)
    lab_float = (colorized_image.astype(np.float32)) 
    lab_float[:, :, 0] *= 100.0 / 255.0
    lab_float[:, :, 1:] -= 128.0
    colorized_image = ImageHelper.lab_to_bgr(lab_float) * 255.0
    return ImageHelper.resize(colorized_image, (size[1], size[0]))