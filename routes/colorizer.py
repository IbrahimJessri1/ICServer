from fastapi import APIRouter, File, UploadFile, status, HTTPException
from repositories import colorizer as repo_colorizer
from models.rgbcolorizer import RGBColorizer
from models.labcolorizer import LabColorizer

colorizer_router = APIRouter(
    prefix="/colorize",
    tags = ['Colorize']
)

@colorizer_router.on_event("startup")
async def initialize_model():
    colorizer_router.rgb_model = RGBColorizer()
    colorizer_router.lab_model = LabColorizer()

@colorizer_router.post('/rgb', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        image_data = await file.read()
        return repo_colorizer.colorize(colorizer_router.rgb_model, image_data, file.content_type)
    else:
        raise HTTPException(status_code=400, detail="File not an image.")
    

@colorizer_router.post('/lab', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        image_data = await file.read()
        return repo_colorizer.colorize(colorizer_router.lab_model, image_data, file.content_type, False)
    else:
        raise HTTPException(status_code=400, detail="File not an image.")