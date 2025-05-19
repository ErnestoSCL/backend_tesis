from fastapi import APIRouter, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.db.models import Evaluacion
from app.schemas.input_data import InputArray
from app.model.data_preprocessor import DataPreprocessor
from app.model.predictor import predecir
from pydantic import EmailStr
from fastapi_mail import FastMail, MessageSchema
from app.utils.email_config import conf  # configuración separada
import tempfile
import shutil
import os
import numpy as np

router = APIRouter()

# Dependencia para obtener una sesión de base de datos
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/predict")
def predict(data: InputArray, db: Session = Depends(get_db)):
    if len(data.values) != 25:
        return {"error": f"Se esperaban 25 valores y se recibieron {len(data.values)}"}

    processor = DataPreprocessor(data.values)
    feature_vector = processor.get_feature_vector()
    resultado = predecir(feature_vector)

    data_dict = processor.get_ordered_column_dict()
    data_dict["rasgos_tea"] = "Si" if resultado["clase_predicha"] == 1 else "No"
    data_dict["nivel_confianza"] = round(resultado["riesgo_autismo"] / 100, 2)

    # Convertir valores numpy a tipos nativos
    for k, v in data_dict.items():
        if isinstance(v, (np.float64, np.float32)):
            data_dict[k] = float(v)
        elif isinstance(v, (np.int64, np.int32)):
            data_dict[k] = int(v)

    evaluacion = Evaluacion(**data_dict)
    db.add(evaluacion)
    db.commit()
    db.refresh(evaluacion)

    return {
        "clase_predicha": resultado["clase_predicha"],
        "riesgo_autismo": resultado["riesgo_autismo"]
    }


@router.post("/enviar-pdf")
async def enviar_pdf(
    file: UploadFile = File(...),
    destinatario: EmailStr = Form(...)
):
    # Crear ruta con nombre original del archivo
    temp_dir = tempfile.gettempdir()
    final_path = os.path.join(temp_dir, file.filename)

    with open(final_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Preparar y enviar
    message = MessageSchema(
        subject="📎 Informe adjunto",
        recipients=[destinatario],
        body="Adjuntamos el informe PDF solicitado.",
        attachments=[final_path],
        subtype="plain"
    )

    fm = FastMail(conf)
    await fm.send_message(message)

    os.remove(final_path)

    return JSONResponse(content={"mensaje": f"Informe de evaluación enviado correctamente a {destinatario}"})
