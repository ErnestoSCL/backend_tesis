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
    feature_vector = processor.get_feature_vector()
    resultado = predecir(feature_vector)

    return {
        "clase_predicha": resultado["clase_predicha"],
        "n_confianza": resultado["n_confianza"],
    }