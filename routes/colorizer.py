from fastapi import APIRouter, File, UploadFile, status, HTTPException
from repositories import colorizer as repo_colorizer
from models.rgbcolorizer import RGBColorizer
from models.labcolorizer import LabColorizer
from utils.realesrgan import RealESRGANEnhancer
from config import colorization_consts
from models.generator13 import Generator
from pathlib import Path

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
        raise HTTPException(status_code=400, detail="Something Went Wrong.")
    
@colorizer_router.post('/gen11', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    try:
        if file.content_type.startswith('image/'):
            image_data = await file.read()
            return repo_colorizer.colorize(colorizer_router.model_gen_11, image_data, file.content_type, False, enhancer = colorizer_router.sr_model)
        else:
            raise HTTPException(status_code=400, detail="File not an image.")
    except:
        raise HTTPException(status_code=400, detail="Something Went Wrong.")
    
@colorizer_router.post('/gen12', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    try:
        if file.content_type.startswith('image/'):
            image_data = await file.read()
            return repo_colorizer.colorize(colorizer_router.model_gen_12, image_data, file.content_type, False, enhancer = colorizer_router.sr_model)
        else:
            raise HTTPException(status_code=400, detail="File not an image.")
    except:
        raise HTTPException(status_code=400, detail="Something Went Wrong.")
    
@colorizer_router.post('/gen13', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    try:
        if file.content_type.startswith('image/'):
            image_data = await file.read()
            return repo_colorizer.colorize(colorizer_router.model_gen_13, image_data, file.content_type, False, enhancer = colorizer_router.sr_model)
        else:
            raise HTTPException(status_code=400, detail="File not an image.")
    except:
        raise HTTPException(status_code=400, detail="Something Went Wrong.")
    

@colorizer_router.post('/gen14', status_code=status.HTTP_200_OK)
async def colorize(file: UploadFile = File(...)):
    try:
        if file.content_type.startswith('image/'):
            image_data = await file.read()
            return repo_colorizer.colorize(colorizer_router.model_gen_14, image_data, file.content_type, False, enhancer = colorizer_router.sr_model)
        else:
            raise HTTPException(status_code=400, detail="File not an image.")
    except:
        raise HTTPException(status_code=400, detail="Something Went Wrong.")
    



# @colorizer_router.get("/test_all_models", status_code=status.HTTP_200_OK)
# async def test_all_models(input_folder: str, output_folder: str):

#     input_path = Path(input_folder)
#     output_path = Path(output_folder)

#     # Check if folders exist
#     if not input_path.is_dir() or not output_path.is_dir():
#         raise HTTPException(status_code=400, detail="Input or output path does not exist.")

#     for image_file in input_path.glob("*"): # this will get all files, you can filter with "*.jpg" or "*.png" if needed
#         if image_file.is_file():
#             with open(image_file, "rb") as f:
#                 image_data = f.read()

#                 for model_suffix in ["gen_8", "gen_11", "gen_12", "gen_13", "gen_14"]:
#                     model = getattr(colorizer_router, f"model_{model_suffix}", None)
#                     if model:
#                         # change output of repo_colorizer.colorize to only BytesIO(...) instead of StreamingResponse
#                         if model_suffix == 'gen_8':
#                             output_image_data = repo_colorizer.colorize(model, image_data, 'image/jpeg', True, enhancer = colorizer_router.sr_model)
#                         else:
#                             output_image_data = repo_colorizer.colorize(model, image_data, 'image/jpeg', False, enhancer = colorizer_router.sr_model)
                            
#                         output_image_bytes = output_image_data.getvalue()

#                         output_filename = output_path / f"{image_file.stem}_{model_suffix}{image_file.suffix}"
#                         with open(output_filename, "wb") as out_f:
#                             out_f.write(output_image_bytes)

    
#     return {"status": "success", "message": f"Processed images saved in {output_folder}"}