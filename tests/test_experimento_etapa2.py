import pandas as pd
import pytest

from src.experimento_etapa2 import (ARQUIVOS_SAIDA, FIGURAS_SAIDA, carregar_entradas, executar,
                                    gerar_dobras, verificar_sem_vazamento)

GRADE_MINIMA = {
    "knn_frac": {"n_neighbors": [3], "weights": ["distance"], "metric": ["euclidean"]},
    "knn_razao": {"metric": ["euclidean"]},
    "svm_rbf": {"C": [1], "gamma": ["scale"]},
    "rf": {"n_estimators": [20], "max_depth": [None]},
}
SEEDS = (42, 43)


def _rodar(execucao, nome, tmp_path_factory):
    saida = tmp_path_factory.mktemp(nome)
    return saida, executar(execucao["dir"] / "seeds", saida, seeds=SEEDS, grade=GRADE_MINIMA)


@pytest.fixture(scope="module")
def rodada(execucao, tmp_path_factory):
    return _rodar(execucao, "etapa2_a", tmp_path_factory)


def test_arquivos_de_saida_criados(rodada):
    saida, _ = rodada
    for arq in ARQUIVOS_SAIDA:
        assert (saida / arq).exists(), arq
    for fig in FIGURAS_SAIDA:
        assert (saida / "figuras" / fig).exists(), fig
    comparacao = pd.read_csv(saida / "etapa2_comparacao.csv")
    assert set(comparacao["modelo"]) == {"knn", "svm", "rf"}
    assert len(comparacao) == 8  # 4 configurações × sem/com escala


def test_mesmas_dobras_para_todos_os_modelos(rodada, execucao):
    saida, res = rodada
    comparacao = pd.read_csv(saida / "etapa2_comparacao.csv")
    assert comparacao["hash_dobras"].nunique() == 1
    semente = res["semente"]
    d1, d2 = gerar_dobras(semente, SEEDS), gerar_dobras(semente, SEEDS)
    for seed in SEEDS:
        for (tr1, va1), (tr2, va2) in zip(d1[seed], d2[seed]):
            assert list(tr1) == list(tr2) and list(va1) == list(va2)


def test_nenhum_genero_em_treino_e_validacao(rodada):
    _, res = rodada
    semente = res["semente"]
    for divisoes in res["dobras"].values():
        for idx_tr, idx_val in divisoes:
            assert not (set(semente.iloc[idx_tr]["genero_atual"])
                        & set(semente.iloc[idx_val]["genero_atual"]))


def test_nenhum_indice_do_teste_bloqueado_ou_oculto_usado(rodada):
    _, res = rodada
    ent = res["entradas"]
    usados = res["indices_usados"]
    oculto = set(ent["gabarito"].loc[ent["gabarito"]["grupo"] == "oculto", "indice"])
    assert usados == set(res["semente"]["indice"])
    assert not usados & set(ent["teste"]["indice"])
    assert not usados & ent["bloqueados_idx"]
    assert not usados & oculto


def test_verificar_sem_vazamento_detecta_injecao(execucao):
    ent = carregar_entradas(execucao["dir"] / "seeds")
    semente, teste = ent["semente"], ent["teste"]
    oculto = ent["gabarito"].loc[ent["gabarito"]["grupo"] == "oculto", "indice"]
    verificar_sem_vazamento(semente, teste["indice"], ent["bloqueados_idx"], oculto,
                            ent["generos_bloqueados"])  # sem vazamento: não levanta
    for intruso in (teste.iloc[[0]], ent["gabarito"][ent["gabarito"]["grupo"] == "oculto"].iloc[[0]]):
        contaminada = pd.concat([semente, intruso], ignore_index=True)
        with pytest.raises(ValueError, match="Vazamento"):
            verificar_sem_vazamento(contaminada, teste["indice"], ent["bloqueados_idx"], oculto,
                                    ent["generos_bloqueados"])


def test_reprodutivel_com_a_mesma_semente(rodada, execucao, tmp_path_factory):
    saida_a, _ = rodada
    saida_b, _ = _rodar(execucao, "etapa2_b", tmp_path_factory)
    for arq in ("etapa2_comparacao.csv", "etapa2_oculto_por_familia.csv",
                "etapa2_oculto_vizinhos.csv", "etapa2_importancias_rf.csv"):
        assert (saida_a / arq).read_bytes() == (saida_b / arq).read_bytes(), arq
