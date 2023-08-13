from fastapi import APIRouter, File, UploadFile, status, HTTPException
from repositories import colorizer as repo_colorizer
from models.rgbcolorizer import RGBColorizer
from models.labcolorizer import LabColorizer
from utils.realesrgan import RealESRGANEnhancer
from config import colorization_consts
from models.generator13 import Generator

colorizer_router = APIRouter(
    prefix="/colorize",
    tags = ['Colorize']
)

@colorizer_router.on_event("startup")
async def initialize_model():
    colorizer_router.sr_model = RealESRGANEnhancer()
    colorizer_router.model_gen_8 = RGBColorizer(colorization_consts.GEN_8_MODEL_PATH)
    colorizer_router.model_gen_11 = LabColorizer(colorization_consts.GEN_11_MODEL_PATH)
    colorizer_router.model_gen_12 = LabColorizer(colorization_consts.GEN_12_MODEL_PATH)
    colorizer_router.model_gen_13 = LabColorizer(colorization_consts.GEN_13_MODEL_PATH, weights_gen_class=Generator)
    colorizer_router.model_gen_14 = LabColorizer(colorization_consts.GEN_14_MODEL_PATH)

@colorizer_router.post('/gen8', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    try:
        if file.content_type.startswith('image/'):
            image_data = await file.read()
            return repo_colorizer.colorize(colorizer_router.model_gen_8, image_data, file.content_type, True, enhancer = colorizer_router.sr_model)
        else:
            raise HTTPException(status_code=400, detail="File not an image.")
    except:
        raise HTTPException(status_code=400, detail="File not an image.")
    
@colorizer_router.post('/gen11', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    try:
        if file.content_type.startswith('image/'):
            image_data = await file.read()
            return repo_colorizer.colorize(colorizer_router.model_gen_11, image_data, file.content_type, False, enhancer = colorizer_router.sr_model)
        else:
            raise HTTPException(status_code=400, detail="File not an image.")
    except:
        raise HTTPException(status_code=400, detail="File not an image.")
    
@colorizer_router.post('/gen12', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    try:
        if file.content_type.startswith('image/'):
            image_data = await file.read()
            return repo_colorizer.colorize(colorizer_router.model_gen_12, image_data, file.content_type, False, enhancer = colorizer_router.sr_model)
        else:
            raise HTTPException(status_code=400, detail="File not an image.")
    except:
        raise HTTPException(status_code=400, detail="File not an image.")
    
@colorizer_router.post('/gen13', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    try:
        if file.content_type.startswith('image/'):
            image_data = await file.read()
            return repo_colorizer.colorize(colorizer_router.model_gen_13, image_data, file.content_type, False, enhancer = colorizer_router.sr_model)
        else:
            raise HTTPException(status_code=400, detail="File not an image.")
    except:
        raise HTTPException(status_code=400, detail="File not an image.")
    

@colorizer_router.post('/gen14', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    try:
        if file.content_type.startswith('image/'):
            image_data = await file.read()
            return repo_colorizer.colorize(colorizer_router.model_gen_14, image_data, file.content_type, False, enhancer = colorizer_router.sr_model)
        else:
            raise HTTPException(status_code=400, detail="File not an image.")
    except:
        raise HTTPException(status_code=400, detail="File not an image.")