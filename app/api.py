from fastapi import APIRouter
from app.schemas.input_data import InputArray

router = APIRouter()

@router.post("/predict")
def predict(data: InputArray):
    campos = [
        "edad", "sexo", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10",
        "ResultadoQCHAT",
        "TrastornoHabla", "TrastornoAprendizaje", "TrastornosGeneticos", "Depresion",
        "RetrasoDesarrollo", "ProblemasSociales", "Ansiedad", "FamiliaConAutismo",
        "DefSocial", "DefComunicativa", "HoraInicio", "HoraFin"
    ]

    if len(data.values) != len(campos):
        return {"error": "Se esperaban 25 valores"}

    data_dict = dict(zip(campos, data.values))
    return data_dict  # solo para probar en Swagger