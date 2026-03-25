import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score


def generar_caso_de_uso_analizar_sensores():
    rng = np.random.default_rng()

    n_rows  = int(rng.integers(20, 41))
    ventana = int(rng.integers(2, 6))

    # Serie de temperatura con anomalías esporádicas
    t           = np.linspace(0, 4 * np.pi, n_rows)
    amplitud    = rng.uniform(5, 20)
    offset      = rng.uniform(300, 500)
    ruido       = rng.standard_normal(n_rows) * rng.uniform(0.5, 2.0)
    temperatura = offset + amplitud * np.sin(t) + ruido

    # Inyectar 1-2 picos bruscos visibles
    n_picos   = int(rng.integers(1, 3))
    idx_picos = rng.choice(n_rows, size=n_picos, replace=False)
    temperatura[idx_picos] += rng.uniform(15, 40, size=n_picos)

    df = pd.DataFrame({"Temperatura": np.round(temperatura, 4)})

    # ── Calcular output esperado ──────────────────────────────────────────────
    df_work = df.copy()
    df_work["Volatilidad"] = df_work["Temperatura"].rolling(window=ventana).std()
    df_work = df_work.dropna().reset_index(drop=True)

    percentil_90 = df_work["Volatilidad"].quantile(0.90)
    y = (df_work["Volatilidad"] > percentil_90).astype(int).values
    X = df_work[["Volatilidad", "Temperatura"]].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    rf     = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    prec   = float(precision_score(y_test, y_pred, zero_division=0))

    return (
        # ── INPUT ────────────────────────────────────────────────────────────
        {
            "df":      df.copy(),
            "ventana": ventana,
        },
        # ── OUTPUT ───────────────────────────────────────────────────────────
        prec,
    )


if __name__ == "__main__":
    i, o = generar_caso_de_uso_analizar_sensores()

    print('---- inputs ----')
    for k, v in i.items():
        print('\n', k, ':\n', v)

    print('\n---- expected output ----\n', o)