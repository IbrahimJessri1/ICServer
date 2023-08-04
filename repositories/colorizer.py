
from models.basecolorizer import BaseColorizer
from utils.imagehelper import ImageHelper
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
import numpy as np
from config import colorization_consts

def colorize(colorizer_model : BaseColorizer, image_data, content_type, rgb = True, enhancer = None):
    if rgb:
        colorized_image = colorize_rgb(colorizer_model, image_data, enhancer=enhancer)
    else:
        colorized_image = colorize_lab(colorizer_model, image_data, enhancer=enhancer)
    try:
        output_image_bytes, media_type = ImageHelper.image_to_output_file(colorized_image, content_type)
    except:
        raise HTTPException(status_code=400, detail="Something went wrong..")
    return StreamingResponse(BytesIO(output_image_bytes), media_type=media_type)

def colorize_rgb(colorizer_model, image_data, enhancer = None):
    image = ImageHelper.read_image(image_data)
    size = image.shape
    input_image = ImageHelper.resize_image(image, (colorization_consts.IMAGE_HEIGHT['rgb'],colorization_consts.IMAGE_WIDTH['rgb']))
    input_image = ImageHelper.bgr_to_gray(input_image)
    norm_input_image = ImageHelper.rgb_normalize(input_image)
    colorized_image = colorizer_model.colorize(norm_input_image)
    colorized_image = ImageHelper.rgb_to_bgr(colorized_image)
    return ImageHelper.resize_image(colorized_image, (size[1], size[0]), enhancer)

def colorize_lab(colorizer_model, image_data, enhancer = None):
    image = ImageHelper.read_image(image_data)
    size = image.shape
    image = ImageHelper.resize_image(image, (colorization_consts.IMAGE_HEIGHT['lab'],colorization_consts.IMAGE_WIDTH['lab']))
    image /= 255.0
    input_image = ImageHelper.bgr_to_lab(image)
    input_image = input_image[:, :, 0]
    if colorizer_model.neg_norm:
        input_image *= 2.0 / 100.0 
        norm_input_image = input_image - 1.0
    else:
        norm_input_image = input_image / 100.0

    colorized_image = colorizer_model.colorize(norm_input_image)
    lab_float = (colorized_image.astype(np.float32)) 
    lab_float[:, :, 0] *= 100.0 / 255.0
    lab_float[:, :, 1:] -= 128.0
    colorized_image = ImageHelper.lab_to_bgr(lab_float) * 255.0
    return ImageHelper.resize_image(colorized_image, (size[1], size[0]), enhancer)