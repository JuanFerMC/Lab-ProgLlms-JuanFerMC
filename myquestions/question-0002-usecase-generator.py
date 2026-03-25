import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


def generar_caso_de_uso_optimizar_dimensiones():
    rng = np.random.default_rng()

    target_col = str(rng.choice(
        ["habito_label", "target", "enfermedad", "clase", "target_variable"]
    ))

    # Reintentar hasta que n_componentes sea válido para el df resultante
    while True:
        n_rows     = int(rng.integers(6, 11))
        n_features = int(rng.integers(3, 6))

        base  = rng.standard_normal((n_rows, max(2, n_features - 1)))
        coef  = rng.standard_normal((base.shape[1], n_features))
        X_raw = base @ coef + rng.standard_normal((n_rows, n_features)) * 0.5

        # Inyectar ~15 % de NaN
        mask = rng.random((n_rows, n_features)) < 0.15
        X_raw[mask] = np.nan

        feature_cols = [f"var_{i}" for i in range(n_features)]
        df = pd.DataFrame(X_raw, columns=feature_cols)
        df[target_col] = rng.integers(0, 2, size=n_rows).tolist()

        df_clean = df.dropna(subset=feature_cols)

        # n_componentes debe ser <= min(n_samples, n_features) y haber >= 2 clases
        max_componentes = min(len(df_clean), n_features)
        if max_componentes >= 2 and df_clean[target_col].nunique() >= 2:
            # Elegir n_componentes ya conociendo el tamaño real del df limpio
            n_componentes = int(rng.integers(2, max_componentes + 1))
            break

    # ── Calcular output esperado ──────────────────────────────────────────────
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca   = PCA(n_components=n_componentes)
    X_pca = pca.fit_transform(X_scaled)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_scaled, y)
    feature_importances = rf.feature_importances_

    return (
        # ── INPUT ────────────────────────────────────────────────────────────
        {
            "df":            df.copy(),
            "target_col":    target_col,
            "n_componentes": n_componentes,
        },
        # ── OUTPUT ───────────────────────────────────────────────────────────
        (X_pca, feature_importances),
    )


if __name__ == "__main__":
    i, o = generar_caso_de_uso_optimizar_dimensiones()

    print('---- inputs ----')
    for k, v in i.items():
        print('\n', k, ':\n', v)

    print('\n---- expected output ----\n', o)
