from typing import Union
from fastapi import FastAPI
from dotenv import load_dotenv
import os

# Cargar el archivo .env
load_dotenv()

# Obtener la variable de entorno
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    raise RuntimeError("DATABASE_URL no está definida en el archivo .env")

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}