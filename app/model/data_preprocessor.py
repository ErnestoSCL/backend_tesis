from pathlib import Path
import joblib
import pandas as pd  # Asegúrate de tenerlo importado arriba

class DataPreprocessor:
    def __init__(self, raw_values):
        self.campos = [
            "Age_Years", "Sex_M", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10_Autism_Spectrum_Quotient",
            "Qchat_10_Score",
            "Speech_Delay_Language_Disorder", "Learning_Disorder", "Genetic_Disorders", "Depression",
            "Global_Developmental_Delay_Intellectual_Disability", "Social_Behavioural_Issues", "Anxiety_Disorder", "Family_Mem_With_Asd",
            "Social_Interaction_Issues_%", "Communication_Issues_%", "Time_Start", "Time_End"
        ]
        self.data = dict(zip(self.campos, raw_values))

        # 🔽 Parámetros min-max para normalización
        self.scaling_params = {
            "Age_Years": (1, 18),
            "Sex_M": (0, 1),
            "A1": (0, 1),
            "A2": (0, 1),
            "A3": (0, 1),
            "A4": (0, 1),
            "A5": (0, 1),
            "A6": (0, 1),
            "A7": (0, 1),
            "A8": (0, 1),
            "A9": (0, 1),
            "A10_Autism_Spectrum_Quotient": (0, 1),
            "Qchat_10_Score": (0, 10),
            "Speech_Delay_Language_Disorder": (0, 1),
            "Learning_Disorder": (0, 1),
            "Genetic_Disorders": (0, 1),
            "Depression": (0, 1),
            "Global_Developmental_Delay_Intellectual_Disability": (0, 1),
            "Social_Behavioural_Issues": (0, 1),
            "Anxiety_Disorder": (0, 1),
            "Family_Mem_With_Asd": (0, 1),
            "Social_Interaction_Issues_%": (0, 100),
            "Communication_Issues_%": (0, 100),
            "Comorbidity_%": (0, 100),
            "Clinical_Profile_Mixed": (0, 1),
            "Clinical_Profile_Social interaction": (0, 1)
        }

    def minmax_scale(self, variable, value):
        min_val, max_val = self.scaling_params[variable]
        if max_val == min_val:
            return 0
        return round((value - min_val) / (max_val - min_val), 4)

    def get_comorbidity_percent(self):
        comorb_vars = [
            "Speech_Delay_Language_Disorder", "Learning_Disorder", "Genetic_Disorders",
            "Depression", "Global_Developmental_Delay_Intellectual_Disability",
            "Social_Behavioural_Issues", "Anxiety_Disorder"
        ]
        total = sum(int(self.data[var]) for var in comorb_vars)
        porcentaje = (total / len(comorb_vars)) * 100
        return round(porcentaje, 2)
    
    def get_clinical_profile(self):
        comm = float(self.data["Communication_Issues_%"])
        social = float(self.data["Social_Interaction_Issues_%"])
        if abs(comm - social) < 10:
            return "mixto", 1, 0
        elif comm > social:
            return "comunicativo", 0, 0
        else:
            return "interactivo-social", 0, 1

    def get_normalized_dataframe(self):
        _, dummy_mixed, dummy_social = self.get_clinical_profile()
        comorb = self.get_comorbidity_percent()

        # Variables extendidas
        variables = {
            **self.data,
            "Comorbidity_%": comorb,
            "Clinical_Profile_Mixed": dummy_mixed,
            "Clinical_Profile_Social interaction": dummy_social
        }

        # Orden usado para entrenar PCA
        ordered_vars = [
            "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10_Autism_Spectrum_Quotient",
            "Age_Years",
            "Qchat_10_Score",
            "Speech_Delay_Language_Disorder", "Learning_Disorder", "Genetic_Disorders", "Depression",
            "Global_Developmental_Delay_Intellectual_Disability", "Social_Behavioural_Issues",
            "Anxiety_Disorder", "Family_Mem_With_Asd",
            "Comorbidity_%",
            "Communication_Issues_%",
            "Social_Interaction_Issues_%",
            "Sex_M",
            "Clinical_Profile_Mixed", "Clinical_Profile_Social interaction"
        ]

        valores = [self.minmax_scale(var, float(variables[var])) for var in ordered_vars]
        return pd.DataFrame([valores], columns=ordered_vars)

    
    def get_pca_component_1(self):
        df_vector = self.get_normalized_dataframe()

        pca_path = Path(__file__).resolve().parent.parent.parent / "pca_model.pkl"
        pca = joblib.load(pca_path)

        return round(pca.transform(df_vector)[0][0], 4)


    def get_feature_vector(self):
        # Obtener PCA_1
        pca1 = self.get_pca_component_1()
        
        # Calcular valores extendidos necesarios
        comorb = self.get_comorbidity_percent()

        # Crear diccionario solo con las variables que se usarán
        variables = {
            **{k: self.data[k] for k in [
                "A6", "A9", "Social_Interaction_Issues_%", "A7", "A5", "Qchat_10_Score",
                "Communication_Issues_%", "A4", "A1", "A2", "A8", "Sex_M", "A3",
                "Global_Developmental_Delay_Intellectual_Disability", "Speech_Delay_Language_Disorder",
                "Depression", "Social_Behavioural_Issues", "Anxiety_Disorder"
            ]},
            "Comorbidity_%": comorb
        }

        # Orden esperado por el modelo AdaBoost
        ordered_vars = [
            "PCA_1", "A6", "A9", "Social_Interaction_Issues_%", "A7", "A5", "Qchat_10_Score",
            "Communication_Issues_%", "A4", "A1", "A2", "A8", "Sex_M", "A3",
            "Global_Developmental_Delay_Intellectual_Disability", "Speech_Delay_Language_Disorder",
            "Depression", "Social_Behavioural_Issues", "Anxiety_Disorder", "Comorbidity_%"
        ]

        # Escalar automáticamente todas las variables excepto PCA_1
        valores = {
            "PCA_1": pca1,
            **{var: self.minmax_scale(var, float(variables[var])) for var in ordered_vars[1:]}
        }

        # Retornar como DataFrame para evitar warnings en el modelo
        return pd.DataFrame([valores], columns=ordered_vars)

    def get_data_dict(self):
        return self.data