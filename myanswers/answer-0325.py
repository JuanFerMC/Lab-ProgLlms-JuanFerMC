import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

# ==========================================
# 1. FUNCIÓN DE SOLUCIÓN
# ==========================================
def detectar_anomalias_sensores(df, contaminacion=0.05):
    """
    Identifica anomalías en registros de sensores y devuelve los puntajes de decisión.
    """
    # Selección de columnas numéricas (presión, temperatura, vibración)
    datos_numericos = df.select_dtypes(include=[np.number])

    # Manejo de valores nulos mediante imputación por mediana
    imputador = SimpleImputer(strategy='median')
    datos_limpios = imputador.fit_transform(datos_numericos)

    # Configuración del modelo Isolation Forest con el nivel de contaminación dado
    modelo = IsolationForest(contamination=contaminacion, random_state=42)
    modelo.fit(datos_limpios)
    
    # Obtención de puntajes de decisión (decision_function)
    # Valores más bajos indican una mayor probabilidad de ser una anomalía.
    scores = modelo.decision_function(datos_limpios)

    return scores.tolist()

# ==========================================
# 2. GENERADOR DE CASOS DE USO (LITERAL)
# ==========================================
def generar_caso_de_uso_detectar_anomalias_industriales(contaminacion=0.05):
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    """
    # Datos de ejemplo
    df = pd.DataFrame({
        "temperatura": [50, 52, 49, 300, 51, 48, 47, 500],
        "presion": [30, 29, 31, 100, 30, 29, 28, 150],
        "vibracion": [0.2, 0.25, 0.22, 5.0, 0.21, 0.19, 0.2, 6.0]
    })

    # Selección numérica
    datos_numericos = df.select_dtypes(include=[np.number])

    # Imputación
    imputador = SimpleImputer(strategy='median')
    datos_limpios = imputador.fit_transform(datos_numericos)

    # Modelo
    modelo = IsolationForest(contamination=contaminacion, random_state=42)

    # Resultados
    predicciones = modelo.fit_predict(datos_limpios)
    scores = modelo.decision_function(datos_limpios)

    # Convertir predicciones a diccionario
    resultado_dict = {
        "anomalias": predicciones.tolist()
    }

    # Retornar lo que pide la plataforma
    return resultado_dict, scores.tolist()

# ==========================================
# 3. EJECUCIÓN Y SALIDA
# ==========================================
if __name__ == "__main__":
    # Generamos los datos de entrada usando el código literal del generador
    input_dict, ground_truth = generar_caso_de_uso_detectar_anomalias_industriales()

    # Los datos del DataFrame y la contaminación esperada (0.05 por defecto)
    df_sensores = pd.DataFrame({
        "temperatura": [50, 52, 49, 300, 51, 48, 47, 500],
        "presion": [30, 29, 31, 100, 30, 29, 28, 150],
        "vibracion": [0.2, 0.25, 0.22, 5.0, 0.21, 0.19, 0.2, 6.0]
    })
    
    # Ejecución de la solución final
    resultado_scores = detectar_anomalias_sensores(df_sensores, contaminacion=0.05)

    print("---- output ----")
    print(resultado_scores)