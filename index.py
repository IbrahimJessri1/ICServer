from fastapi import FastAPI
from routes import colorizer

app = FastAPI()

app.include_router(colorizer.colorizer_router)
