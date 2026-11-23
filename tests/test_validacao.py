import warnings

import pandas as pd
import pytest

from src.validacao import validacao_agrupada


def _sem_genero_compartilhado(df, divisoes):
    for idx_treino, idx_val in divisoes:
        treino = set(df.iloc[idx_treino]["genero_atual"])
        val = set(df.iloc[idx_val]["genero_atual"])
        assert not treino & val


def test_validacao_agrupada_no_treino_inicial(execucao):
    treino = execucao["treino"].reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # o fixture pode ter poucos gêneros positivos
        divisoes = validacao_agrupada(treino)
    assert len(divisoes) >= 2
    _sem_genero_compartilhado(treino, divisoes)


def test_reduz_n_splits_com_aviso():
    df = pd.DataFrame({
        "genero_atual": ["A", "A", "B", "C", "C", "D", "E", "F"],
        "rotulo": [1, 1, 1, 1, 1, 0, 0, 0],
    })
    with pytest.warns(UserWarning, match="reduzido de 5 para 3"):
        divisoes = validacao_agrupada(df, n_splits=5)
    assert len(divisoes) == 3
    _sem_genero_compartilhado(df, divisoes)


def test_erro_com_um_genero_positivo():
    df = pd.DataFrame({"genero": ["A", "B", "C"], "rotulo": [1, 0, 0]})
    with pytest.raises(ValueError, match="ao menos 2"):
        validacao_agrupada(df)
