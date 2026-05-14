import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer


def detectar_anomalias_industriales(input_dict: dict, df: pd.DataFrame, contaminacion: float = 0.05) -> list:
    """
    Calcula los scores de decisión de Isolation Forest para cada registro.

    Args:
        input_dict (dict): Diccionario con clave "anomalias" (lista de 1/-1 ya calculadas).
        df (pd.DataFrame): DataFrame original con las lecturas de los sensores.
        contaminacion (float): Porcentaje esperado de anomalías en los datos (ej. 0.05).

    Returns:
        list: Scores de decisión por registro (valores positivos = normal, negativos = anomalía).
    """
    # Seleccionar solo columnas numéricas
    datos_numericos = df.select_dtypes(include=[np.number])

    # Imputar valores faltantes con la mediana
    imputador = SimpleImputer(strategy='median')
    datos_limpios = imputador.fit_transform(datos_numericos)

    # Entrenar el modelo con los mismos parámetros que el generador
    modelo = IsolationForest(contamination=contaminacion, random_state=42)
    modelo.fit(datos_limpios)

    # Retornar los scores de decisión (el expected output del generador)
    return modelo.decision_function(datos_limpios).tolist()


def generar_caso_de_uso_detectar_anomalias_industriales(contaminacion=0.05):
    
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

    # 🔥 Convertir predicciones a diccionario
    resultado_dict = {
        "anomalias": predicciones.tolist()
    }

    # ✅ Retornar lo que pide la plataforma
    return resultado_dict, scores.tolist()

if __name__ == "__main__":
    i, o = generar_caso_de_uso_detectar_anomalias_industriales()

    print('---- inputs ----')
    for k, v in i.items():
        print('\n', k, ':\n', v)

    print('\n---- expected output ----\n', o)

    # ── Verificación de la solución ──
    df_test = pd.DataFrame({
        "temperatura": [50, 52, 49, 300, 51, 48, 47, 500],
        "presion": [30, 29, 31, 100, 30, 29, 28, 150],
        "vibracion": [0.2, 0.25, 0.22, 5.0, 0.21, 0.19, 0.2, 6.0]
    })

    resultado = detectar_anomalias_industriales(i, df_test)

    print('\n---- solution output ----\n', resultado)
    print('\n---- match ----\n', resultado == o)