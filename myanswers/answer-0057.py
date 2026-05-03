import numpy as np
from sklearn.decomposition import PCA
import random

def calcular_componentes_pca(X, varianza_minima):
    """
    Calcula el número mínimo de componentes necesarios para explicar 
    una proporción determinada de la varianza total.
    """
    # 1. Ajustar el modelo PCA sobre la matriz X
    # No fijamos n_components para que calcule todos los posibles
    pca = PCA()
    pca.fit(X)

    # 2. Calcular la varianza explicada acumulada
    # explained_variance_ratio_ nos da la varianza por cada componente individual
    var_acum = np.cumsum(pca.explained_variance_ratio_)

    # 3. Identificar el número mínimo de componentes
    # np.searchsorted encuentra el primer índice donde la varianza acumulada 
    # es mayor o igual al umbral especificado.
    n_componentes = np.searchsorted(var_acum, varianza_minima) + 1

    # 4. Retornar el número de componentes como entero
    return int(n_componentes)

def generar_caso_de_uso_calcular_componentes_pca():
    """
    Genera un caso aleatorio para la función calcular_componentes_pca
    """

    # 1. Dimensiones aleatorias
    n_samples = random.randint(20, 80)
    n_features = random.randint(4, 10)

    # 2. Datos correlacionados
    base = np.random.randn(n_samples, 2)
    ruido = np.random.randn(n_samples, n_features - 2) * 0.2
    X = np.hstack([base @ np.random.randn(2, n_features - 2), ruido])

    # 3. Varianza mínima aleatoria
    varianza_minima = random.uniform(0.6, 0.95)

    input_data = {
        "X": X.copy(),
        "varianza_minima": varianza_minima
    }

    # OUTPUT esperado
    pca = PCA()
    pca.fit(X)
    var_acum = np.cumsum(pca.explained_variance_ratio_)
    n_componentes = np.searchsorted(var_acum, varianza_minima) + 1

    output_data = n_componentes

    return input_data, output_data

if __name__ == "__main__":    # Ejemplo de uso
    input_data, output_data = generar_caso_de_uso_calcular_componentes_pca()

    X = input_data["X"]
    varianza_minima = input_data["varianza_minima"]
    
    n_componentes = calcular_componentes_pca(X, varianza_minima)
    print(f"Número de componentes necesarios para explicar al menos {varianza_minima*100}% de la varianza: {n_componentes}")
