from fastapi import APIRouter, File, UploadFile, status, HTTPException
from repositories import colorizer as repo_colorizer
from models.rgbcolorizer import RGBColorizer
from models.labcolorizer import LabColorizer
from utils.realesrgan import RealESRGANEnhancer
from config import colorization_consts

colorizer_router = APIRouter(
    prefix="/colorize",
    tags = ['Colorize']
)

@colorizer_router.on_event("startup")
async def initialize_model():
    colorizer_router.sr_model = RealESRGANEnhancer()
    colorizer_router.rgb_model = RGBColorizer()
    colorizer_router.lab_model1 = LabColorizer(colorization_consts.LAB_MODEL_PATH_1, neg_norm= False)
    colorizer_router.lab_model2 = LabColorizer(colorization_consts.LAB_MODEL_PATH_2)
    colorizer_router.lab_model4 = LabColorizer(colorization_consts.LAB_MODEL_PATH_4)
    colorizer_router.lab_model5 = LabColorizer(colorization_consts.LAB_MODEL_PATH_5)
    colorizer_router.lab_model6_1 = LabColorizer(colorization_consts.LAB_MODEL_PATH_6_1, weights=True)
    colorizer_router.lab_model6_2 = LabColorizer(colorization_consts.LAB_MODEL_PATH_6_2, weights=True)
    colorizer_router.lab_model6_3 = LabColorizer(colorization_consts.LAB_MODEL_PATH_6_3, weights=True)
    colorizer_router.lab_model6_4 = LabColorizer(colorization_consts.LAB_MODEL_PATH_6_4, weights=True)
    colorizer_router.lab_model6_5 = LabColorizer(colorization_consts.LAB_MODEL_PATH_6_5, weights=True)
    colorizer_router.lab_model6_6 = LabColorizer(colorization_consts.LAB_MODEL_PATH_6_6, weights=True)

@colorizer_router.post('/rgb', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        image_data = await file.read()
        return repo_colorizer.colorize(colorizer_router.rgb_model, image_data, file.content_type, True, enhancer = colorizer_router.sr_model)
    else:
        raise HTTPException(status_code=400, detail="File not an image.")
    

@colorizer_router.post('/lab1', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        image_data = await file.read()
        return repo_colorizer.colorize(colorizer_router.lab_model1, image_data, file.content_type, False, enhancer = colorizer_router.sr_model)
    else:
        raise HTTPException(status_code=400, detail="File not an image.")
    
@colorizer_router.post('/lab2', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        image_data = await file.read()
        return repo_colorizer.colorize(colorizer_router.lab_model2, image_data, file.content_type, False, enhancer = colorizer_router.sr_model)
    else:
        raise HTTPException(status_code=400, detail="File not an image.")
    
@colorizer_router.post('/lab4', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        image_data = await file.read()
        return repo_colorizer.colorize(colorizer_router.lab_model4, image_data, file.content_type, False, enhancer = colorizer_router.sr_model)
    else:
        raise HTTPException(status_code=400, detail="File not an image.")
    
@colorizer_router.post('/lab5', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        image_data = await file.read()
        return repo_colorizer.colorize(colorizer_router.lab_model5, image_data, file.content_type, False, enhancer = colorizer_router.sr_model)
    else:
        raise HTTPException(status_code=400, detail="File not an image.")
    
@colorizer_router.post('/lab6_9', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        image_data = await file.read()
        return repo_colorizer.colorize(colorizer_router.lab_model6_1, image_data, file.content_type, False, enhancer = colorizer_router.sr_model)
    else:
        raise HTTPException(status_code=400, detail="File not an image.")
    
@colorizer_router.post('/lab6_15', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        image_data = await file.read()
        return repo_colorizer.colorize(colorizer_router.lab_model6_2, image_data, file.content_type, False, enhancer = colorizer_router.sr_model)
    else:
        raise HTTPException(status_code=400, detail="File not an image.")

@colorizer_router.post('/lab6_21', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        image_data = await file.read()
        return repo_colorizer.colorize(colorizer_router.lab_model6_3, image_data, file.content_type, False, enhancer = colorizer_router.sr_model)
    else:
        raise HTTPException(status_code=400, detail="File not an image.")
    
@colorizer_router.post('/lab6_25', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        image_data = await file.read()
        return repo_colorizer.colorize(colorizer_router.lab_model6_4, image_data, file.content_type, False, enhancer = colorizer_router.sr_model)
    else:
        raise HTTPException(status_code=400, detail="File not an image.")
    

@colorizer_router.post('/lab6_39', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        image_data = await file.read()
        return repo_colorizer.colorize(colorizer_router.lab_model6_5, image_data, file.content_type, False, enhancer = colorizer_router.sr_model)
    else:
        raise HTTPException(status_code=400, detail="File not an image.")
    
@colorizer_router.post('/lab6_40', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        image_data = await file.read()
        return repo_colorizer.colorize(colorizer_router.lab_model6_6, image_data, file.content_type, False, enhancer = colorizer_router.sr_model)
    else:
        raise HTTPException(status_code=400, detail="File not an image.")