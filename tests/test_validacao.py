import warnings

import pandas as pd
import pytest

from src.validacao import avaliar_candidatos, validacao_agrupada


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


def _gabarito_sintetico():
    return pd.DataFrame({
        "indice": [0, 1, 2, 3, 4, 5, 6],
        "rotulo_gabarito": [1, 1, 1, 0, 0, 1, 1],
        "grupo": ["oculto", "oculto", "oculto", "oculto", "semente", "semente", "teste"],
    })


def test_avaliar_candidatos_acumulado():
    res = avaliar_candidatos([3, 0, 99, 5, 1], _gabarito_sintetico())
    assert list(res["no_pool"]) == [True, True, False, False, True]
    assert list(res["morcegos_reais_acum"]) == [0, 1, 1, 1, 2]
    assert list(res["candidatos_no_pool_acum"]) == [1, 2, 2, 2, 3]
    assert res["precisao_acum"].tolist() == pytest.approx([0, 0.5, 0.5, 0.5, 2 / 3])
    assert res["recall_acum"].tolist() == pytest.approx([0, 1 / 3, 1 / 3, 1 / 3, 2 / 3])
    assert list(res["fora_do_pool_acum"]) == [0, 0, 1, 1, 1]


def test_avaliar_candidatos_recusa_teste_e_duplicados():
    with pytest.raises(ValueError, match="vazamento"):
        avaliar_candidatos([0, 6], _gabarito_sintetico())
    with pytest.raises(ValueError, match="repetidos"):
        avaliar_candidatos([0, 0], _gabarito_sintetico())


def test_avaliar_candidatos_no_gabarito_real(execucao):
    gabarito = pd.read_csv(execucao["dir"] / "seeds" / "gabarito.csv")
    oculto = gabarito[gabarito["grupo"] == "oculto"]
    morcegos = oculto.loc[oculto["rotulo_gabarito"] == 1, "indice"]
    res = avaliar_candidatos(morcegos, gabarito)
    assert res["recall_acum"].iloc[-1] == pytest.approx(1.0)
    assert (res["precisao_acum"] == 1.0).all()
