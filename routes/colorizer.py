from fastapi import APIRouter, File, UploadFile, status, HTTPException
from repositories import colorizer as repo_colorizer
from models.rgbcolorizer import RGBColorizer
from models.labcolorizer import LabColorizer
from config import colorization_consts

colorizer_router = APIRouter(
    prefix="/colorize",
    tags = ['Colorize']
)

@colorizer_router.on_event("startup")
async def initialize_model():
    colorizer_router.rgb_model = RGBColorizer()
    colorizer_router.lab_model1 = LabColorizer(colorization_consts.LAB_MODEL_PATH_1)
    colorizer_router.lab_model2 = LabColorizer(colorization_consts.LAB_MODEL_PATH_2)
    colorizer_router.lab_model4 = LabColorizer(colorization_consts.LAB_MODEL_PATH_4)
    colorizer_router.lab_model5 = LabColorizer(colorization_consts.LAB_MODEL_PATH_5)
    colorizer_router.lab_model6 = LabColorizer(colorization_consts.LAB_MODEL_PATH_6, True)

@colorizer_router.post('/rgb', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        image_data = await file.read()
        return repo_colorizer.colorize(colorizer_router.rgb_model, image_data, file.content_type)
    else:
        raise HTTPException(status_code=400, detail="File not an image.")
    

@colorizer_router.post('/lab1', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        image_data = await file.read()
        return repo_colorizer.colorize(colorizer_router.lab_model1, image_data, file.content_type, False)
    else:
        raise HTTPException(status_code=400, detail="File not an image.")
    
@colorizer_router.post('/lab2', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        image_data = await file.read()
        return repo_colorizer.colorize(colorizer_router.lab_model2, image_data, file.content_type, False)
    else:
        raise HTTPException(status_code=400, detail="File not an image.")
    
@colorizer_router.post('/lab4', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        image_data = await file.read()
        return repo_colorizer.colorize(colorizer_router.lab_model4, image_data, file.content_type, False)
    else:
        raise HTTPException(status_code=400, detail="File not an image.")
    
@colorizer_router.post('/lab5', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        image_data = await file.read()
        return repo_colorizer.colorize(colorizer_router.lab_model5, image_data, file.content_type, False)
    else:
        raise HTTPException(status_code=400, detail="File not an image.")
    
@colorizer_router.post('/lab6', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        image_data = await file.read()
        return repo_colorizer.colorize(colorizer_router.lab_model6, image_data, file.content_type, False)
    else:
        raise HTTPException(status_code=400, detail="File not an image.")