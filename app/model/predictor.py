import joblib
from pathlib import Path

def predecir(df):
    # Cargar el modelo desde la raíz del proyecto
    model_path = Path(__file__).resolve().parent.parent.parent / "model.pkl"
    model = joblib.load(model_path)

    # Obtener la probabilidad de clase 1
    probas = model.predict_proba(df)[0]
    prob_clase_1 = probas[1]

    # Aplicar umbral personalizado
    umbral = 0.605
    clase_predicha = int(prob_clase_1 >= umbral)

    # Convertir probabilidad a porcentaje (solo la de la clase predicha)
    probabilidad = probas[clase_predicha]
    riesgo_autismo = round(probabilidad * 100, 2)

    return {
        "clase_predicha": clase_predicha,
        "riesgo_autismo": riesgo_autismo,
    }