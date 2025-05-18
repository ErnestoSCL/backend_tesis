from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app import api
# Para crear las tablas en Railway (solo la primera vez o si no existen)
from app.db.models import Base
from app.db.database import engine

# Crear automáticamente las tablas si no existen
Base.metadata.create_all(bind=engine)

app = FastAPI()
# CORS para permitir acceso desde React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api.router)