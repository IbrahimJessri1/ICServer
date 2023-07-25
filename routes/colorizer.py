from fastapi import APIRouter, File, UploadFile, status, HTTPException
from repositories import colorizer as repo_colorizer
from models.colorizer import Colorizer

colorizer_router = APIRouter(
    prefix="/colorize",
    tags = ['Colorize']
)

# @colorizer_router.on_event("startup")
# async def initialize_model():
#     colorizer_router.model = Colorizer()

colorizer_router.model = None

@colorizer_router.post('/', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        image_data = await file.read()
        return repo_colorizer.colorize(colorizer_router.model, image_data, file.content_type)
    else:
        raise HTTPException(status_code=400, detail="File not an image.")