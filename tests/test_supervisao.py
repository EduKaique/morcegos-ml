import hashlib
import warnings

import pandas as pd
import pytest

from src.carregar_dados import N_ESPERADO
from src.supervisao import (
    ListaTaxonomicaAusente,
    anotar_taxonomia,
    carregar_taxonomia,
    executar,
    familias_elegiveis,
)

INDICES_SUSPEITOS = [8422, 8423, 8424, 8425]


def _ler(execucao, nome):
    return pd.read_csv(execucao["dir"] / "seeds" / nome)


def test_arquivos_gerados(execucao):
    seeds = execucao["dir"] / "seeds"
    for nome in ["positivos.csv", "negativos.csv", "treino_inicial.csv", "teste_verificado.csv",
                 "casos_ambiguos.csv", "teste_verificado.sha256"]:
        assert (seeds / nome).exists(), nome
    assert (execucao["dir"] / "results" / "relatorio_supervisao.md").exists()


def test_rotulos_validos(execucao):
    rotulo = execucao["anotado"]["rotulo_fonte"]
    assert set(rotulo.dropna().unique()) <= {0, 1}
    for nome in ["positivos.csv", "negativos.csv", "treino_inicial.csv", "teste_verificado.csv"]:
        assert set(_ler(execucao, nome)["rotulo"].unique()) <= {0, 1}, nome


def test_indices_unicos_no_intervalo(execucao):
    for nome in ["positivos.csv", "negativos.csv", "treino_inicial.csv", "teste_verificado.csv"]:
        indice = _ler(execucao, nome)["indice"]
        assert indice.is_unique, nome
        assert indice.between(0, N_ESPERADO - 1).all(), nome


def test_treino_e_teste_disjuntos(execucao):
    treino, teste = _ler(execucao, "treino_inicial.csv"), _ler(execucao, "teste_verificado.csv")
    assert not set(treino["indice"]) & set(teste["indice"])
    assert not set(treino["genero_atual"]) & set(teste["genero_atual"])
    assert not set(treino["genero"]) & set(teste["genero"])


def test_teste_sem_ambiguos_nem_suspeitos(execucao):
    anotado = execucao["anotado"].set_index("indice")
    teste = _ler(execucao, "teste_verificado.csv")
    assert not anotado.loc[teste["indice"], "ambiguo"].any()
    assert not anotado.loc[teste["indice"], "genero_suspeito"].any()


def test_suspeitos_nunca_rotulados(execucao):
    anotado = execucao["anotado"].set_index("indice")
    assert anotado.loc[INDICES_SUSPEITOS, "genero_suspeito"].all()
    assert anotado.loc[INDICES_SUSPEITOS, "rotulo_fonte"].isna().all()
    assert not set(INDICES_SUSPEITOS) & set(execucao["rotulados"]["indice"])


def test_familias_elegiveis_no_teste(execucao):
    elegiveis = familias_elegiveis(execucao["rotulados"])
    assert elegiveis, "nenhuma família elegível: o teste não estaria exercitando a regra"
    familias_teste = set(execucao["teste"].query("rotulo == 1")["familia"])
    for familia in elegiveis:
        assert familia in familias_teste, familia


def test_sha256_confere(execucao):
    seeds = execucao["dir"] / "seeds"
    gravado = (seeds / "teste_verificado.sha256").read_text().split()[0]
    assert gravado == hashlib.sha256((seeds / "teste_verificado.csv").read_bytes()).hexdigest()


def test_reexecucao_igual_e_deterministica(execucao, taxonomia):
    """Rodar de novo com o mesmo seed gera o mesmo teste e não acusa congelamento."""
    executar(*taxonomia, execucao["dir"] / "seeds", execucao["dir"] / "results")


def test_teste_congelado_nao_e_sobrescrito(execucao, taxonomia, tmp_path):
    seeds = tmp_path / "seeds"
    executar(*taxonomia, seeds, tmp_path / "results")
    with pytest.raises(RuntimeError, match="congelado"):
        executar(*taxonomia, seeds, tmp_path / "results", seed=7)
    executar(*taxonomia, seeds, tmp_path / "results", seed=7, sobrescrever_teste=True)


def test_lista_vazia_nao_grava_nada(tmp_path):
    vazia = tmp_path / "generos_mamiferos.csv"
    vazia.write_text("genero,ordem,familia,fonte,data_acesso\n")
    with pytest.raises(ListaTaxonomicaAusente):
        executar(vazia, tmp_path / "sinonimos.csv", tmp_path / "seeds", tmp_path / "results")
    assert not (tmp_path / "seeds").exists()


def test_taxonomia_colunas_e_duplicados(tmp_path):
    arq = tmp_path / "tax.csv"
    arq.write_text("genero,ordem\nMyotis,Chiroptera\n")
    with pytest.raises(ValueError, match="faltam as colunas"):
        carregar_taxonomia(arq)
    arq.write_text("genero,ordem,familia,fonte,data_acesso\n"
                   " Myotis ,Chiroptera,F1,x,2026-01-01\n"
                   "Myotis,Chiroptera,F2,x,2026-01-01\n")
    with pytest.warns(UserWarning, match="Myotis"):
        tax = carregar_taxonomia(arq)
    assert tax.to_dict("records")[0]["genero"] == "Myotis"
    assert len(tax) == 1


def test_sinonimo_aplicado():
    df = pd.DataFrame({"indice": [0, 1], "nome": ["Hylomys_suillus", "Abc_def"],
                       "genero": ["Hylomys", "Abc"]})
    tax = pd.DataFrame({"genero": ["Erinaceus"], "ordem": ["Eulipotyphla"], "familia": ["F"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = anotar_taxonomia(df, tax, {"Hylomys": "Erinaceus"})
    assert out.loc[0, "genero_atual"] == "Erinaceus"
    assert out.loc[0, "rotulo_fonte"] == 0
    assert pd.isna(out.loc[1, "rotulo_fonte"])
