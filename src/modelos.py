"""Classificadores da Etapa 2: k-NN, SVM RBF e Random Forest.

Todos são montados como ``sklearn.pipeline.Pipeline`` (escala opcional +
modelo), para que o ``StandardScaler`` seja ajustado só no treino de cada
dobra. Todos têm a mesma interface de pontuação (``pontuar``): um valor maior
significa "mais provável morcego".
"""

import itertools

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
COLUNAS = [f"c{i:02d}" for i in range(50)]

MODELOS = ("knn", "svm", "rf")

GRADE_PADRAO = {
    "knn_frac": {"n_neighbors": [1, 3, 5, 7], "weights": ["uniform", "distance"],
                 "metric": ["euclidean", "manhattan", "cosine"]},
    "knn_razao": {"metric": ["euclidean", "manhattan", "cosine"]},
    "svm_rbf": {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1]},
    "rf": {"n_estimators": [100, 300], "max_depth": [None, 4, 8]},
}

# variante -> modelo
VARIANTES = {"knn_frac": "knn", "knn_razao": "knn", "svm_rbf": "svm", "rf": "rf"}


class RazaoDistanciaKNN(ClassifierMixin, BaseEstimator):
    """Pontuação ``d_neg / (d_pos + d_neg)`` pelas distâncias ao vizinho mais próximo.

    ``d_pos`` e ``d_neg`` são as distâncias ao positivo e ao negativo mais
    próximos do treino recebido em ``fit`` (um ``NearestNeighbors`` por classe).
    Vale 1 sobre um positivo, 0 sobre um negativo e 0.5 a igual distância.
    """

    def __init__(self, metric="euclidean"):
        self.metric = metric

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        self.vizinhos_pos_ = NearestNeighbors(n_neighbors=1, metric=self.metric,
                                              algorithm="brute").fit(X[y == 1])
        self.vizinhos_neg_ = NearestNeighbors(n_neighbors=1, metric=self.metric,
                                              algorithm="brute").fit(X[y == 0])
        return self

    def decision_function(self, X):
        d_pos = self.vizinhos_pos_.kneighbors(X)[0][:, 0]
        d_neg = self.vizinhos_neg_.kneighbors(X)[0][:, 0]
        soma = d_pos + d_neg
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(soma > 0, d_neg / np.where(soma > 0, soma, 1), 0.5)

    def predict(self, X):
        return (self.decision_function(X) > 0.5).astype(int)


def _estimador(variante, params):
    if variante == "knn_frac":
        return KNeighborsClassifier(algorithm="brute", **params)
    if variante == "knn_razao":
        return RazaoDistanciaKNN(**params)
    if variante == "svm_rbf":
        return SVC(kernel="rbf", class_weight="balanced", **params)
    if variante == "rf":
        return RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE,
                                      n_jobs=-1, **params)
    raise ValueError(f"Variante desconhecida: {variante}")


def construir_pipeline(variante, params, escala):
    """Pipeline ``escala`` (``StandardScaler`` ou ``passthrough``) + ``modelo``."""
    return Pipeline([("escala", StandardScaler() if escala else "passthrough"),
                     ("modelo", _estimador(variante, params))])


def pontuar(pipe, X):
    """Pontuação do pipeline ajustado: maior = mais provável morcego.

    Usa ``decision_function`` quando existe (SVM, razão de distâncias); senão
    ``predict_proba`` da classe positiva (k-NN por fração de vizinhos, RF).
    """
    modelo = pipe[-1]
    if hasattr(modelo, "decision_function"):
        return np.asarray(pipe.decision_function(X), dtype=float)
    coluna = list(modelo.classes_).index(1)
    return pipe.predict_proba(X)[:, coluna]


def _rotulo(nome, valor):
    abrev = {"n_neighbors": "k", "weights": "w", "metric": "m", "n_estimators": "n",
             "max_depth": "prof"}
    return f"{abrev.get(nome, nome)}={valor}"


def gerar_configuracoes(grade=GRADE_PADRAO):
    """Lista de configurações ``{modelo, variante, escala, params, config_id}``.

    Cada combinação da grade aparece sem e com ``StandardScaler``.
    """
    configs = []
    for variante, opcoes in grade.items():
        nomes = list(opcoes)
        for valores in itertools.product(*(opcoes[n] for n in nomes)):
            params = dict(zip(nomes, valores))
            for escala in (False, True):
                partes = [variante] + [_rotulo(n, v) for n, v in params.items()]
                partes.append("escala" if escala else "sem_escala")
                configs.append({"modelo": VARIANTES[variante], "variante": variante,
                                "escala": escala, "params": params,
                                "config_id": "|".join(partes)})
    return configs
