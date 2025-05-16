from fastapi import APIRouter
from app.schemas.input_data import InputArray
from app.model.data_preprocessor import DataPreprocessor
from app.model.predictor import predecir

router = APIRouter()

@router.post("/predict")
def predict(data: InputArray):
    if len(data.values) != 25:
        return {"error": f"Se esperaban 25 valores y se recibieron {len(data.values)}"}

    processor = DataPreprocessor(data.values)

    # Obtener predicción y datos
    feature_vector = processor.get_feature_vector()
    resultado = predecir(feature_vector)
    datos_originales = processor.data  # ← Diccionario con las 25 variables originales

    return {
        "riesgo_autismo": resultado["riesgo_autismo"],
        "clase_predicha": resultado["clase_predicha"],
        "vector_entrada": feature_vector.values.tolist()[0],
        "datos_originales": datos_originales,
        "mensaje": "Evaluación procesada correctamente"
    }