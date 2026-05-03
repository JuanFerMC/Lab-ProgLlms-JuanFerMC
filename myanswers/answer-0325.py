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
    # Selección de columnas numéricas
    datos_numericos = df.select_dtypes(include=[np.number])

    # Imputación por mediana para asegurar que no haya valores nulos
    imputador = SimpleImputer(strategy='median')
    datos_limpios = imputador.fit_transform(datos_numericos)

    # Entrenamiento del modelo Isolation Forest
    modelo = IsolationForest(contamination=contaminacion, random_state=42)
    modelo.fit(datos_limpios)
    
    # El decision_function devuelve el score de anomalía (más bajo = más anómalo)
    scores = modelo.decision_function(datos_limpios)

    return scores.tolist()

# ==========================================
# 2. GENERADOR DE CASOS DE USO (MODIFICADO)
# ==========================================
def generar_caso_de_uso_detectar_anomalias_industriales(contaminacion=0.05):
    """
    Genera los datos de prueba y calcula los resultados esperados.
    Se modifica para retornar el DataFrame original y evitar doble creación.
    """
    # Creación única del DataFrame de sensores
    df = pd.DataFrame({
        "temperatura": [50, 52, 49, 300, 51, 48, 47, 500],
        "presion": [30, 29, 31, 100, 30, 29, 28, 150],
        "vibracion": [0.2, 0.25, 0.22, 5.0, 0.21, 0.19, 0.2, 6.0]
    })

    # Procesamiento interno para obtener el ground truth (scores)
    datos_numericos = df.select_dtypes(include=[np.number])
    imputador = SimpleImputer(strategy='median')
    datos_limpios = imputador.fit_transform(datos_numericos)

    modelo = IsolationForest(contamination=contaminacion, random_state=42)
    modelo.fit(datos_limpios)
    scores = modelo.decision_function(datos_limpios)

    # ✅ Retornamos el DF y los scores calculados
    return df, scores.tolist()

# ==========================================
# 3. EJECUCIÓN UNIFICADA
# ==========================================
if __name__ == "__main__":
    # Llamamos al generador una sola vez para obtener los datos y el resultado esperado
    i, expected_output = generar_caso_de_uso_detectar_anomalias_industriales(contaminacion=0.05)

    # Pasamos el DataFrame generado directamente a la función de solución
    resultado_final = detectar_anomalias_sensores(i, contaminacion=0.05)

    print("---- output ----")
    print(resultado_final)