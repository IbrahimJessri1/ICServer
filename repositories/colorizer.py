
from models.colorizer import Colorizer
from utils.imagehelper import ImageHelper
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO

def colorize(colorizer_model : Colorizer, image_data, content_type):
    image = ImageHelper.read_image(image_data)
    input_image = ImageHelper.bgr_to_gray(image)
    size = input_image.shape
    input_image = ImageHelper.resize_input(input_image)
    norm_input_image = ImageHelper.rgb_normalize(input_image)
    colorized_image = colorizer_model.colorize(norm_input_image)
    colorized_image = ImageHelper.rgb_to_bgr(colorized_image)
    colorized_image = ImageHelper.resize(colorized_image, (size[1], size[0]))
    try:
        output_image_bytes, media_type = ImageHelper.image_to_output_file(colorized_image, content_type)
    except:
        raise HTTPException(status_code=400, detail="Something went wrong..")
    return StreamingResponse(BytesIO(output_image_bytes), media_type=media_type)


