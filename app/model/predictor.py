import joblib
from pathlib import Path

def predecir(df):
    # Cargar el modelo desde la raíz del proyecto
    model_path = Path(__file__).resolve().parent.parent.parent / "model.pkl"
    model = joblib.load(model_path)

    # Predecir clase (0 o 1) y obtener todas las probabilidades
    clase_predicha = int(model.predict(df)[0])
    probabilidades = model.predict_proba(df)[0]

    # Obtener la probabilidad correspondiente a la clase predicha
    probabilidad = probabilidades[clase_predicha]

    # Convertir a porcentaje redondeado
    riesgo_autismo = round(probabilidad * 100, 2)

    return {
        "clase_predicha": clase_predicha,
        "riesgo_autismo": riesgo_autismo
    }
