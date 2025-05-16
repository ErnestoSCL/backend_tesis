import joblib
from pathlib import Path

def predecir(df):
    # Cargar el modelo desde la raíz del proyecto
    model_path = Path(__file__).resolve().parent.parent.parent / "model.pkl"
    model = joblib.load(model_path)

    # Predecir la probabilidad de clase 1 (riesgo)
    clase_predicha = int(model.predict(df)[0])
    probabilidad = model.predict_proba(df)[0][1]  # Clase positiva

    # Convertir a porcentaje redondeado
    riesgo = round(probabilidad * 100, 2)

    return {
        "riesgo_autismo": riesgo,
        "clase_predicha": clase_predicha
    }
