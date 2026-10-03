"""
Vector de referencia condicionado a la hoja, específico para Random Forest.

En vez de una referencia global fija (media/mediana/cero, igual para
cualquier instancia), calcula por instancia un valor de referencia por
celda (t, v) a partir de los ejemplos de entrenamiento que caen en la
misma hoja que la instancia en cada árbol del ensamble. Evita forzar
al modelo a extrapolar fuera de la región que sus propios árboles
consideraron al construir la predicción (Hooker, Mentch y Zhou, 2021).

La primera versión promediaba todos los ejemplos co-hoja por igual.
Eso no considera que la pertenencia a la hoja depende de la ventana
completa (T*V celdas), mientras que la oclusión solo reemplaza una
celda a la vez: dos ventanas pueden compartir hoja y aun así diferir
arbitrariamente en el resto de sus celdas. Esta versión pondera cada
ejemplo co-hoja por su cercanía a la ventana completa que se está
explicando (kernel gaussiano, ancho de banda adaptativo = mediana de
distancias dentro de la hoja, sin hiperparámetro fijo que ajustar),
para que la referencia sea coherente tanto con la hoja como con el
resto del contexto de la instancia.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def leaf_conditional_reference(rf_model: RandomForestRegressor,
                                x_flat: np.ndarray,
                                X_train_flat: np.ndarray,
                                T: int,
                                V: int,
                                min_neighbors: int = 5) -> np.ndarray:
    """
    Parámetros
    ----------
    rf_model : RandomForestRegressor ya entrenado (sklearn).
    x_flat : np.ndarray, shape (T*V,)
        Instancia a explicar, aplanada.
    X_train_flat : np.ndarray, shape (N_train, T*V)
        Datos de entrenamiento, aplanados, usados para construir rf_model.
    T, V : int
        Dimensiones de la ventana original antes de aplanar.
    min_neighbors : int
        Si la hoja tiene menos ejemplos co-hoja que este número, se usa
        el promedio simple de la hoja en vez del kernel (evita un ancho
        de banda inestable con muestras muy pequeñas).

    Retorna
    -------
    reference : np.ndarray, shape (T, V)
        Valor de referencia por celda, promediado sobre los árboles.
    """
    x_flat = x_flat.astype(np.float32)
    x_2d = x_flat.reshape(1, -1)
    acc = np.zeros(T * V, dtype=np.float64)
    global_mean = X_train_flat.mean(axis=0)  # fallback si una hoja queda vacía

    for tree in rf_model.estimators_:
        leaf_id = tree.apply(x_2d)[0]
        train_leaves = tree.apply(X_train_flat.astype(np.float32))
        mask = train_leaves == leaf_id
        n_co = int(mask.sum())

        if n_co == 0:
            local_ref = global_mean
        else:
            X_leaf = X_train_flat[mask]
            dists = np.linalg.norm(X_leaf - x_flat, axis=1)
            if n_co < min_neighbors:
                local_ref = X_leaf.mean(axis=0)
            else:
                bandwidth = np.median(dists)
                if bandwidth < 1e-8:
                    local_ref = X_leaf.mean(axis=0)
                else:
                    weights = np.exp(-(dists ** 2) / (2.0 * bandwidth ** 2))
                    weights_sum = weights.sum()
                    if weights_sum < 1e-12:
                        local_ref = X_leaf.mean(axis=0)
                    else:
                        weights /= weights_sum
                        local_ref = (weights[:, None] * X_leaf).sum(axis=0)
        acc += local_ref

    acc /= len(rf_model.estimators_)
    return acc.reshape(T, V).astype(np.float32)


def leaf_conditional_reference_batch(rf_model: RandomForestRegressor,
                                      X_flat: np.ndarray,
                                      X_train_flat: np.ndarray,
                                      T: int,
                                      V: int) -> np.ndarray:
    """Aplica leaf_conditional_reference a un lote de instancias.

    Retorna
    -------
    np.ndarray, shape (N, T, V)
    """
    return np.stack([
        leaf_conditional_reference(rf_model, X_flat[i], X_train_flat, T, V)
        for i in range(X_flat.shape[0])
    ])
