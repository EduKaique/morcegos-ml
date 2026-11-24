import hashlib
import warnings

import pandas as pd
import pytest

from src.carregar_dados import N_ESPERADO
from src.supervisao import (
    ListaTaxonomicaAusente,
    anotar_taxonomia,
    carregar_taxonomia,
    GRUPO_OCULTO,
    GRUPO_SEMENTE,
    GRUPO_TESTE,
    executar,
    familias_elegiveis,
    instancias_bloqueadas,
)

INDICES_SUSPEITOS = [8422, 8423, 8424, 8425]


def _ler(execucao, nome):
    return pd.read_csv(execucao["dir"] / "seeds" / nome)


def test_arquivos_gerados(execucao):
    seeds = execucao["dir"] / "seeds"
    for nome in ["positivos.csv", "negativos.csv", "treino_inicial.csv", "teste_verificado.csv",
                 "casos_ambiguos.csv", "teste_verificado.sha256", "gabarito.csv",
                 "semente_reduzida.csv", "generos_bloqueados.csv"]:
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
    elegiveis = familias_elegiveis(execucao["rotulados"], execucao["n_especies"])
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


# --------------------------------------------------------------------------
# Modo semente reduzida
# --------------------------------------------------------------------------

def test_grupos_disjuntos_por_genero(execucao):
    gabarito = _ler(execucao, "gabarito.csv")
    assert set(gabarito["grupo"]) <= {GRUPO_SEMENTE, GRUPO_OCULTO, GRUPO_TESTE}
    assert gabarito.groupby("genero_atual")["grupo"].nunique().max() == 1
    rotulados = execucao["rotulados"]
    assert rotulados.groupby("genero_atual")["grupo"].nunique().max() == 1


def test_gabarito_cobre_todos_os_mamiferos(execucao):
    gabarito = _ler(execucao, "gabarito.csv")
    anotado = execucao["anotado"]
    assert set(gabarito["indice"]) == set(anotado.loc[anotado["rotulo_fonte"].notna(), "indice"])
    assert set(gabarito["rotulo_gabarito"]) <= {0, 1}


def test_nenhum_genero_grande_no_teste(execucao):
    n_especies = execucao["n_especies"]
    teste = _ler(execucao, "teste_verificado.csv")
    assert (teste["genero_atual"].map(n_especies) <= 9).all()
    bloqueados = _ler(execucao, "generos_bloqueados.csv")
    assert (bloqueados["n_instancias_names"] < 10).all()
    assert "Myotis" not in set(bloqueados["genero_atual"])


def test_teste_completado_ate_min_positivos(execucao):
    teste = _ler(execucao, "teste_verificado.csv")
    n_pos = int((teste["rotulo"] == 1).sum())
    assert n_pos >= 5 or any("completar o teste" in a for a in execucao["avisos"])


def test_semente_sem_oculto_nem_teste(execucao):
    semente = _ler(execucao, "semente_reduzida.csv")
    gabarito = _ler(execucao, "gabarito.csv")
    proibidos = set(gabarito.loc[gabarito["grupo"] != GRUPO_SEMENTE, "indice"])
    assert not set(semente["indice"]) & proibidos
    bloqueadas = instancias_bloqueadas(execucao["anotado"], execucao["dir"] / "seeds" / "generos_bloqueados.csv")
    assert not set(semente["indice"]) & set(bloqueadas)
    assert set(semente["rotulo"]) == {0, 1}


def test_treino_inicial_igual_semente(execucao):
    seeds = execucao["dir"] / "seeds"
    assert (seeds / "treino_inicial.csv").read_bytes() == (seeds / "semente_reduzida.csv").read_bytes()


def test_bloqueados_sao_os_generos_do_teste(execucao):
    bloqueados = set(_ler(execucao, "generos_bloqueados.csv")["genero_atual"])
    assert set(_ler(execucao, "teste_verificado.csv")["genero_atual"]) == bloqueados


def test_instancias_bloqueadas_cobre_todo_genero_do_teste(execucao):
    anotado = execucao["anotado"]
    bloqueados = set(_ler(execucao, "generos_bloqueados.csv")["genero_atual"])
    esperado = set(anotado.loc[anotado["genero_atual"].isin(bloqueados), "indice"])
    obtido = instancias_bloqueadas(anotado, execucao["dir"] / "seeds" / "generos_bloqueados.csv")
    assert set(obtido) == esperado
    assert set(_ler(execucao, "teste_verificado.csv")["indice"]) <= set(obtido)


def test_instancias_bloqueadas_inclui_sem_rotulo(tmp_path):
    arq = tmp_path / "generos_bloqueados.csv"
    arq.write_text("genero_atual,origem,ordem,familia,n_instancias_names,n_no_teste\n"
                   "Peixe,negativo_provavel,,,3,1\nAnoura,chiroptera_por_genero,Chiroptera,F,1,1\n")
    df = pd.DataFrame({"indice": [0, 1, 2, 3, 4],
                       "genero": ["Peixe", "Peixe", "Peixe", "Anoura", "Outro"],
                       "genero_atual": ["Peixe", "Peixe", "Peixe", "Anoura", "Outro"],
                       "rotulo_fonte": pd.array([pd.NA, pd.NA, pd.NA, 1, pd.NA], dtype="Int64")})
    assert instancias_bloqueadas(df, arq) == [0, 1, 2, 3]
    # sem genero_atual: usa genero (sinônimos de data/external)
    assert instancias_bloqueadas(df.drop(columns="genero_atual"), arq) == [0, 1, 2, 3]


def test_fracao_semente_nao_muda_teste(execucao, taxonomia, tmp_path):
    executar(*taxonomia, tmp_path / "seeds", tmp_path / "results", fracao_semente=0.6)
    original = (execucao["dir"] / "seeds" / "teste_verificado.sha256").read_text()
    assert (tmp_path / "seeds" / "teste_verificado.sha256").read_text() == original
    semente = pd.read_csv(tmp_path / "seeds" / "semente_reduzida.csv")
    assert len(semente) > len(_ler(execucao, "semente_reduzida.csv"))


def test_positivos_negativos_com_grupo(execucao):
    for nome in ["positivos.csv", "negativos.csv"]:
        assert set(_ler(execucao, nome)["grupo"]) <= {GRUPO_SEMENTE, GRUPO_OCULTO, GRUPO_TESTE}, nome


def test_eulipotyphla_na_semente(execucao):
    semente = _ler(execucao, "semente_reduzida.csv")
    if "Eulipotyphla" in set(execucao["anotado"]["ordem"].dropna()):
        assert "Eulipotyphla" in set(semente["ordem"].dropna())
