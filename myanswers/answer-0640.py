import numpy as np
import pandas as pd
from sklearn.manifold import Isomap

import numpy as np
import pandas as pd
from sklearn.manifold import Isomap

def reducir_movilidad_con_isomap(df, n_componentes, n_neighbors):
    """
    Reduce la dimensionalidad de datos de movilidad urbana conservando 
    relaciones no lineales mediante Isomap.
    """
    # 1. Filtrar únicamente las columnas numéricas
    X = df.select_dtypes(include=[np.number])

    # 2. Configurar el modelo Isomap con los parámetros especificados
    # n_neighbors define cuántos puntos cercanos se consideran para construir el grafo
    # n_components define la dimensión final deseada
    modelo = Isomap(
        n_components=n_componentes,
        n_neighbors=n_neighbors
    )

    # 3. Ajustar y transformar los datos
    # El resultado es un np.ndarray con la representación reducida
    representacion_reducida = modelo.fit_transform(X)

    return representacion_reducida

def generar_caso_de_uso_reducir_movilidad_con_isomap():
    rng = np.random.default_rng()

    n_filas = int(rng.integers(12, 20))
    n_componentes = int(rng.integers(2, 4))
    n_neighbors = int(rng.integers(3, 6))

    df = pd.DataFrame({
        "flujo_vehicular": rng.normal(200, 40, n_filas),
        "velocidad_media": rng.normal(35, 8, n_filas),
        "tiempo_espera": rng.normal(12, 3, n_filas),
        "ocupacion_vial": rng.normal(0.65, 0.1, n_filas)
    })

    input_data = {
        "df": df.copy(),
        "n_componentes": n_componentes,
        "n_neighbors": n_neighbors
    }

    X = df.select_dtypes(include=[np.number])

    modelo = Isomap(
        n_components=n_componentes,
        n_neighbors=n_neighbors
    )
    output_data = modelo.fit_transform(X)

    return input_data, output_data

if __name__ == "__main__":
    i, o = generar_caso_de_uso_reducir_movilidad_con_isomap()

    expt_output = reducir_movilidad_con_isomap(i['df'], i['n_componentes'], i['n_neighbors'])
    print('\n---- output ----\n', expt_output) 