import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def generar_caso_de_uso_optimizar_dimensiones():
    rng = np.random.default_rng()

    target_col = str(rng.choice(
        ["label", "target", "clase", "diagnosis", "target_variable"]
    ))

    # Reintentar hasta obtener suficientes filas limpias para PCA
    while True:
        n_rows     = int(rng.integers(4, 9))
        n_features = int(rng.integers(3, 7))

        # Datos con estructura correlacionada para que PCA sea significativo
        base  = rng.standard_normal((n_rows, max(2, n_features - 1)))
        coef  = rng.standard_normal((base.shape[1], n_features))
        X_raw = base @ coef + rng.standard_normal((n_rows, n_features)) * 0.3

        # Inyectar ~20 % de NaN
        mask = rng.random((n_rows, n_features)) < 0.20
        X_raw[mask] = np.nan

        feature_cols = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X_raw, columns=feature_cols)
        df[target_col] = rng.integers(0, 2, size=n_rows).tolist()

        df_clean = df.dropna(subset=feature_cols)

        # PCA necesita al menos n_features + 1 filas (n_samples - 1 > 0)
        # y al menos 2 clases para que el ejemplo sea interesante
        if len(df_clean) >= n_features + 1 and df_clean[target_col].nunique() >= 2:
            break

    # ── Calcular output esperado ──────────────────────────────────────────────
    X = df_clean[feature_cols].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_full = PCA()
    pca_full.fit(X_scaled)
    var_acum     = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(var_acum, 0.95) + 1)
    n_components = min(n_components, X_scaled.shape[1])

    pca           = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X_scaled)

    return (
        # ── INPUT ────────────────────────────────────────────────────────────
        {
            "df":         df.copy(),
            "target_col": target_col,
        },
        # ── OUTPUT ───────────────────────────────────────────────────────────
        (pca, X_transformed, n_components),
    )


if __name__ == "__main__":
    i, o = generar_caso_de_uso_optimizar_dimensiones()

    print('---- inputs ----')
    for k, v in i.items():
        print('\n', k, ':\n', v)

    print('\n---- expected output ----\n', o)
