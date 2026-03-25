import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def generar_caso_de_uso_segmentar_usuarios():
    rng = np.random.default_rng()

    while True:
        n_rows     = int(rng.integers(6, 11))
        n_features = int(rng.integers(3, 6))

        X_raw = rng.standard_normal((n_rows, n_features))

        # Inyectar ~20 % de NaN para que dropna() elimine algunas filas
        mask = rng.random((n_rows, n_features)) < 0.20
        X_raw[mask] = np.nan

        feature_cols = [f"comportamiento_{i}" for i in range(n_features)]
        df = pd.DataFrame(X_raw, columns=feature_cols)

        df_clean = df.dropna().reset_index(drop=True)

        # Elegir n_componentes y n_clusters ya conociendo el tamaño real
        max_componentes = min(len(df_clean), n_features)
        if max_componentes >= 2 and len(df_clean) >= 2:
            n_componentes = int(rng.integers(2, max_componentes + 1))
            n_clusters    = int(rng.integers(2, min(4, len(df_clean) + 1)))
            break

    # ── Calcular output esperado ──────────────────────────────────────────────
    pca   = PCA(n_components=n_componentes)
    X_pca = pca.fit_transform(df_clean.values)

    kmeans    = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    etiquetas = kmeans.fit_predict(X_pca)

    df_clean["Segmento"] = etiquetas

    return (
        # ── INPUT ────────────────────────────────────────────────────────────
        {
            "df":            df.copy(),
            "n_componentes": n_componentes,
            "n_clusters":    n_clusters,
        },
        # ── OUTPUT ───────────────────────────────────────────────────────────
        df_clean,
    )


if __name__ == "__main__":
    i, o = generar_caso_de_uso_segmentar_usuarios()

    print('---- inputs ----')
    for k, v in i.items():
        print('\n', k, ':\n', v)

    print('\n---- expected output ----\n', o)