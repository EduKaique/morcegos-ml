import numpy as np
import pandas as pd
import pytest

from src.experimento_etapa2 import avaliar_configuracao, metricas_repeticao
from src.modelos import (GRADE_PADRAO, RazaoDistanciaKNN, construir_pipeline,
                         gerar_configuracoes, pontuar)


def test_razao_distancia_extremos():
    X = np.array([[0.0, 0.0], [10.0, 0.0], [11.0, 0.0]])
    y = np.array([1, 0, 0])
    est = RazaoDistanciaKNN().fit(X, y)
    s = est.decision_function(np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 0.0]]))
    assert s == pytest.approx([1.0, 0.0, 0.5])


def test_pontuacao_maior_e_mais_provavel_morcego():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(3, 0.3, (10, 4)), rng.normal(0, 0.3, (30, 4))])
    y = np.array([1] * 10 + [0] * 30)
    for config in gerar_configuracoes({"knn_frac": {"n_neighbors": [3], "weights": ["distance"],
                                                    "metric": ["euclidean"]},
                                       "knn_razao": {"metric": ["cosine"]},
                                       "svm_rbf": {"C": [1], "gamma": ["scale"]},
                                       "rf": {"n_estimators": [20], "max_depth": [None]}}):
        pipe = construir_pipeline(config["variante"], config["params"], config["escala"]).fit(X, y)
        s = pontuar(pipe, np.array([[3.0] * 4, [0.0] * 4]))
        assert s[0] > s[1], config["config_id"]


def test_grade_padrao_tem_os_tres_modelos_com_e_sem_escala():
    configs = gerar_configuracoes(GRADE_PADRAO)
    assert {c["modelo"] for c in configs} == {"knn", "svm", "rf"}
    assert len({c["config_id"] for c in configs}) == len(configs)
    assert sum(c["escala"] for c in configs) == len(configs) / 2


def test_scaler_ajustado_so_no_treino_da_dobra():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(12, 3)) + np.arange(12)[:, None]  # médias bem diferentes por dobra
    y = np.array([1, 0, 0] * 4)
    dobras = {42: [(np.arange(0, 6), np.arange(6, 12)), (np.arange(6, 12), np.arange(0, 6))]}
    config = {"variante": "knn_frac", "escala": True,
              "params": {"n_neighbors": 1, "weights": "uniform", "metric": "euclidean"}}
    _, modelos = avaliar_configuracao(config, X, y, dobras, devolver_modelos=True)
    for pipe, idx_tr in modelos[42]:
        media = pipe.named_steps["escala"].mean_
        assert media == pytest.approx(X[idx_tr].mean(axis=0))
        assert not np.allclose(media, X.mean(axis=0))


def test_metricas_repeticao_limiar_p95():
    y = np.array([1, 1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
    m = metricas_repeticao(y, s)
    assert m["limiar_p95"] == pytest.approx(0.8)  # menor limiar com precisão 1
    assert m["recall_p95"] == pytest.approx(2 / 3)
    assert m["prec"] == pytest.approx(1.0)
    assert m["sens"] == pytest.approx(2 / 3)
    assert m["acc"] == pytest.approx(5 / 6)
    sem = metricas_repeticao(np.array([0, 1]), np.array([0.9, 0.1]))
    assert np.isnan(sem["limiar_p95"]) and sem["sem_limiar_p95"] == 1
