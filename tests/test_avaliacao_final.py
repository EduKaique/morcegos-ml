import json
import shutil

import numpy as np
import pandas as pd
import pytest

import src.avaliacao_final as av
from src.avaliacao_final import (ARQ_LOG, ARQUIVOS_SAIDA, FIGURAS_SAIDA, carregar_base_final,
                                 executar, ler_teste_verificado, verificar_base, wilson)
from src.enriquecimento import SUBDIR_AUTO, carregar_contexto, dir_rodada, executar_auto

N_BOOT = 200


@pytest.fixture(scope="module")
def dirs(execucao, tmp_path_factory):
    """Sementes da Etapa 1 em tmp + rodadas do modo automático (convergidas) em tmp."""
    dir_seeds = execucao["dir"] / "seeds"
    dir_rounds = tmp_path_factory.mktemp("rounds_etapa4")
    executar_auto(dir_rounds, carregar_contexto(dir_seeds), max_rodadas=10)
    return dir_seeds, dir_rounds / SUBDIR_AUTO


def _executar(dirs, dir_res, **kw):
    dir_seeds, dir_rounds = dirs
    return executar(dir_seeds, dir_rounds, dir_res, n_bootstrap=N_BOOT, **kw)


@pytest.fixture(scope="module")
def avaliacao(dirs, tmp_path_factory):
    dir_res = tmp_path_factory.mktemp("res_etapa4_a")
    return dir_res, _executar(dirs, dir_res, avaliar_teste=True)


# --------------------------------------------------------------------------
# Wilson
# --------------------------------------------------------------------------

def test_wilson():
    p, inf, sup = wilson(5, 5)
    assert p == 1 and inf == pytest.approx(0.5655, abs=1e-3) and sup == 1.0
    p, inf, sup = wilson(0, 10)
    assert p == 0 and inf == 0 and sup == pytest.approx(0.2775, abs=1e-3)
    p, inf, sup = wilson(184, 189)
    assert inf < p < sup
    assert all(np.isnan(v) for v in wilson(0, 0))


# --------------------------------------------------------------------------
# Proteções do teste e da base
# --------------------------------------------------------------------------

def test_sha256_confere_e_para_se_alterado(dirs, tmp_path):
    dir_seeds, dir_rounds = dirs
    teste, h = ler_teste_verificado(dir_seeds)
    assert len(teste) > 0 and len(h) == 64
    copia = tmp_path / "seeds"
    shutil.copytree(dir_seeds, copia)
    arq = copia / "teste_verificado.csv"
    arq.write_bytes(arq.read_bytes().replace(b",1,", b",0,", 1))
    with pytest.raises(RuntimeError, match="SHA256"):
        ler_teste_verificado(copia)
    with pytest.raises(RuntimeError, match="SHA256"):
        executar(copia, dir_rounds, tmp_path / "res", avaliar_teste=True, n_bootstrap=N_BOOT)
    assert not (tmp_path / "res" / ARQ_LOG).exists()


def test_base_sem_teste_nem_bloqueados(avaliacao):
    _, r = avaliacao
    base = set(r["base"]["indice"])
    assert not base & set(r["teste"]["indice"])
    assert not base & set(r["todas"].loc[r["todas"]["bloqueada"], "indice"])


def test_base_com_indice_bloqueado_gera_erro(avaliacao):
    _, r = avaliacao
    todas = r["todas"]
    bloq = set(todas.loc[todas["bloqueada"], "indice"])
    intrusa = pd.concat([r["base"], pd.DataFrame([{"indice": min(bloq), "rotulo": 0,
                                                   "origem": "gabarito", "rodada": 9}])])
    ann = todas[["indice"]].assign(genero_atual="x")
    with pytest.raises(ValueError, match="Vazamento"):
        verificar_base(intrusa, ann, bloq, set())


def test_rodada_nao_convergida_gera_erro(dirs, tmp_path):
    _, dir_rounds = dirs
    copia = tmp_path / "rounds"
    shutil.copytree(dir_rounds, copia)
    ultima = max(int(p.name.split("_")[1]) for p in copia.glob("rodada_*"))
    arq = dir_rodada(copia, ultima) / "metricas.json"
    met = json.loads(arq.read_text(encoding="utf-8"))
    met["convergido"] = False
    arq.write_text(json.dumps(met), encoding="utf-8")
    with pytest.raises(RuntimeError, match="não convergiu"):
        carregar_base_final(copia)


def test_sem_flag_nao_le_teste(dirs, tmp_path, monkeypatch):
    def proibido(*a, **kw):
        raise AssertionError("o teste não pode ser lido sem --avaliar-teste")

    leitura_original = pd.read_csv

    def read_csv_vigiado(caminho, *a, **kw):
        if "teste_verificado" in str(caminho):
            proibido()
        return leitura_original(caminho, *a, **kw)

    monkeypatch.setattr(av, "ler_teste_verificado", proibido)
    monkeypatch.setattr(pd, "read_csv", read_csv_vigiado)
    dir_res = tmp_path / "res"
    r = _executar(dirs, dir_res, avaliar_teste=False)
    assert not r["avaliou_teste"] and r["resumo"]["base_pos"] > 0
    assert not dir_res.exists() or not any(dir_res.iterdir())


# --------------------------------------------------------------------------
# Saídas
# --------------------------------------------------------------------------

def test_saidas_criadas(avaliacao):
    dir_res, _ = avaliacao
    for arq in ARQUIVOS_SAIDA + (ARQ_LOG,):
        assert (dir_res / arq).exists(), arq
    for fig in FIGURAS_SAIDA:
        assert (dir_res / "figuras" / fig).exists(), fig


def test_matriz_confusao_soma_teste(avaliacao):
    dir_res, r = avaliacao
    m = pd.read_csv(dir_res / "etapa4_matriz_confusao_teste.csv")
    total = int(m[["previsto_morcego", "previsto_outro"]].to_numpy().sum())
    assert total == len(r["teste"])
    if len(r["teste"]) == 189:  # taxonomia real
        assert int(m.loc[0, ["previsto_morcego", "previsto_outro"]].sum()) == 5


def test_cv_sem_genero_em_treino_e_validacao(avaliacao):
    _, r = avaliacao
    grupos = r["conjunto"]["genero_atual"].to_numpy()
    assert len(r["dobras"]) == 10
    for divisoes in r["dobras"].values():
        for idx_tr, idx_val in divisoes:
            assert not set(grupos[idx_tr]) & set(grupos[idx_val])
    cv = r["cv"]
    assert set(cv["repeticao"]) == {str(s) for s in range(42, 52)} | {"media", "dp"}


def test_logo_cobre_todos_os_generos_de_morcego(avaliacao):
    _, r = avaliacao
    conj = r["conjunto"]
    assert set(r["logo"]["genero"]) == set(conj.loc[conj["rotulo"] == 1, "genero_atual"])
    assert r["logo"]["n"].sum() == (conj["rotulo"] == 1).sum()


def test_pontuacoes_todas(avaliacao):
    _, r = avaliacao
    t = r["todas"]
    assert len(t) == 8426 and list(t["indice"]) == list(range(8426))
    assert t["na_base"].sum() == len(r["base"]) and t["no_teste"].sum() == len(r["teste"])
    assert not (t["na_base"] & t["bloqueada"]).any()
    assert t.loc[t["no_teste"], "bloqueada"].all()
    assert t["rotulo_verificado"].notna().sum() == len(r["base"]) + len(r["teste"])


def test_log_uso_teste(avaliacao, dirs):
    dir_res, _ = avaliacao
    log = pd.read_csv(dir_res / ARQ_LOG)
    assert len(log) >= 1 and list(log.columns) == av.COLUNAS_LOG
    antes = len(log)
    with pytest.warns(UserWarning, match="invalida o teste"):
        _executar(dirs, dir_res, avaliar_teste=True)
    assert len(pd.read_csv(dir_res / ARQ_LOG)) == antes + 1


def test_reprodutivel(avaliacao, dirs, tmp_path):
    dir_a, _ = avaliacao
    dir_b = tmp_path / "res_b"
    _executar(dirs, dir_b, avaliar_teste=True)
    for arq in ARQUIVOS_SAIDA:
        assert (dir_a / arq).read_bytes() == (dir_b / arq).read_bytes(), arq
