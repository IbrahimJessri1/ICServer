
from models.colorizer import Colorizer
from utils.imagehelper import ImageHelper
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO


def colorize(colorizer_model, image_data, content_type):
    image = ImageHelper.read_image(image_data)
    colorized_image = image
    try:
        output_image_bytes, media_type = ImageHelper.image_to_output_file(colorized_image, content_type)
    except:
        raise HTTPException(status_code=400, detail="Something went wrong..")
    return StreamingResponse(BytesIO(output_image_bytes), media_type=media_type)


