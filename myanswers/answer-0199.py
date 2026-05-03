import pandas as pd
import random
import numpy as np

def top_productos_por_categoria(df, n):
    """
    Calcula los productos más rentables por categoría basándose en el ingreso total.
    """
    # 1. Crear la columna ingreso_total
    df_resultado = df.copy()
    df_resultado['ingreso_total'] = df_resultado['precio_unitario'] * df_resultado['cantidad_vendida']

    # 2. Agrupar por categoría y producto para sumar ingresos
    # Usamos as_index=False para mantener las columnas agrupadas como columnas normales
    agrupado = df_resultado.groupby(['categoria', 'producto'], as_index=False)['ingreso_total'].sum()

    # 3. Ordenar por categoría (A-Z) e ingreso (Mayor a Menor)
    agrupado = agrupado.sort_values(['categoria', 'ingreso_total'], ascending=[True, False])

    # 4. Seleccionar el top n por cada categoría
    top_n = agrupado.groupby('categoria').head(n).reset_index(drop=True)

    # 5. Retornar solo las columnas requeridas
    return top_n[['categoria', 'producto', 'ingreso_total']]

def generar_caso_de_uso_top_productos_por_categoria():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función top_productos_por_categoria.
    """

    # ---------------------------------------------------------
    # 1. Generar datos aleatorios
    # ---------------------------------------------------------
    n_rows = random.randint(8, 20)
    productos = [f'producto_{i}' for i in range(1, 8)]
    categorias = ['A', 'B', 'C']
    df = pd.DataFrame({
        'producto': np.random.choice(productos, size=n_rows),
        'categoria': np.random.choice(categorias, size=n_rows),
        'precio_unitario': np.round(np.random.uniform(5, 100, size=n_rows), 2),
        'cantidad_vendida': np.random.randint(1, 20, size=n_rows),
        'fecha': pd.to_datetime(
            np.random.choice(pd.date_range("2023-01-01", "2023-12-31"), size=n_rows)
        )
    })

    # Elegimos n aleatorio (pero válido)
    n = random.randint(1, 3)

    # ---------------------------------------------------------
    # 2. Construir INPUT
    # ---------------------------------------------------------
    input_data = {
        'df': df.copy(),
        'n': n
    }

    # ---------------------------------------------------------
    # 3. Calcular OUTPUT esperado (ground truth)
    # ---------------------------------------------------------
    df_calc = df.copy()
    # A. Calcular ingreso_total
    df_calc['ingreso_total'] = df_calc['precio_unitario'] * df_calc['cantidad_vendida']
    # B. Agrupar por categoria y producto
    grouped = df_calc.groupby(['categoria', 'producto'], as_index=False)['ingreso_total'].sum()
    # C. Ordenar dentro de cada categoría
    grouped = grouped.sort_values(['categoria', 'ingreso_total'], ascending=[True, False]) # type: ignore
    # D. Tomar top n por categoría
    resultado = grouped.groupby('categoria').head(n).reset_index(drop=True)
    output_data = resultado[['categoria', 'producto', 'ingreso_total']]

    return input_data, output_data

if __name__ == "__main__":
    i, o = generar_caso_de_uso_top_productos_por_categoria()
    df = i['df']
    resultado = top_productos_por_categoria(df, 2)
    print("Output: ", resultado)