import json

import pandas as pd
import pytest

import src.enriquecimento as enr
from src.enriquecimento import (ORIGENS_VERIFICADAS, ErroDecisaoHumana, carregar_contexto,
                                dir_rodada, executar_auto, executar_rodada, gerar_saidas,
                                verificar_bloqueio)
from src.exploracao import projetar_pca

MAX_RODADAS = 10


def _pca(X):
    return projetar_pca(X)[0]


@pytest.fixture(scope="module")
def ctx(execucao):
    return carregar_contexto(execucao["dir"] / "seeds")


@pytest.fixture(scope="module")
def auto(ctx, tmp_path_factory):
    dir_rounds = tmp_path_factory.mktemp("rounds")
    dir_res = tmp_path_factory.mktemp("results")
    hist = executar_auto(dir_rounds, ctx, max_rodadas=MAX_RODADAS)
    gerar_saidas(ctx, dir_rounds, dir_res, projetar=_pca)
    return dir_rounds, dir_res, hist


def _rodadas(dir_modo):
    return sorted(p for p in dir_modo.glob("rodada_*") if p.is_dir())


def _ler(caminho):
    return pd.read_csv(caminho, keep_default_na=False, dtype={"decisao_humana": str})


def _rodada_manual_com_revisao(ctx, tmp_path, minimo=1):
    dir_rounds = tmp_path / "rounds"
    executar_rodada(1, dir_rounds, ctx)
    rev = _ler(dir_rodada(dir_rounds, 1) / "revisao_manual.csv")
    if len(rev) < minimo:
        pytest.skip(f"a taxonomia usada gera só {len(rev)} candidato(s) sem rótulo")
    return dir_rounds, rev


def _gravar_decisoes(dir_rounds, rev, decisoes):
    rev = rev.copy()
    for pos, valor in decisoes.items():
        rev.loc[pos, "decisao_humana"] = valor
    rev.to_csv(dir_rodada(dir_rounds, 1) / "revisao_manual.csv", index=False)


# --------------------------------------------------------------------------
# Modo automático
# --------------------------------------------------------------------------

def test_auto_converge_e_para(auto):
    dir_rounds, _, hist = auto
    assert 1 <= len(hist) <= MAX_RODADAS
    ultima = hist[-1]
    assert ultima["convergido"] or "limite" in ultima["motivo"]
    if ultima["convergido"]:
        assert ultima["positivos_novos"] == 0
    assert len(_rodadas(dir_rounds / "auto")) == len(hist)
    assert not _rodadas(dir_rounds)  # nada nas rodadas manuais


def test_nenhum_indice_bloqueado_em_candidatos_ou_base(auto, ctx):
    dir_rounds, _, _ = auto
    for d in _rodadas(dir_rounds / "auto"):
        for arq in ("candidatos.csv", "base.csv", "revisao_manual.csv"):
            indices = set(_ler(d / arq)["indice"].astype(int))
            assert not indices & ctx["proibidos"], (d.name, arq)


def test_base_so_cresce_com_origem_verificada(auto, ctx):
    dir_rounds, _, _ = auto
    anterior = set(ctx["semente"]["indice"].astype(int))
    for d in _rodadas(dir_rounds / "auto"):
        base = _ler(d / "base.csv")
        atual = set(base["indice"])
        assert anterior <= atual
        novas = base[base["rodada"] > 0]
        assert novas["origem"].isin(ORIGENS_VERIFICADAS).all()
        # no automático só o gabarito do oculto entra
        assert set(novas["indice"]) <= ctx["oculto"]
        anterior = atual


def test_pendentes_sem_rotulo_nao_entram_no_auto(auto):
    dir_rounds, _, _ = auto
    for d in _rodadas(dir_rounds / "auto"):
        cand = _ler(d / "candidatos.csv")
        pend = set(cand.loc[cand["verificacao"] != "gabarito", "indice"])
        assert not pend & set(_ler(d / "base.csv")["indice"])


def test_saidas_criadas(auto):
    _, dir_res, hist = auto
    assert (dir_res / "relatorio_etapa3.md").exists()
    historico = pd.read_csv(dir_res / "etapa3_historico.csv")
    assert len(historico) == len(hist) and set(historico["modo"]) == {"auto"}
    for fig in ("etapa3_positivos_por_rodada.png", "etapa3_recall_familia_por_rodada.png",
                "etapa3_umap.png"):
        assert (dir_res / "figuras" / fig).exists()


def test_avaliar_candidatos_recebe_so_oculto(ctx, tmp_path, monkeypatch):
    chamadas = []
    original = enr.avaliar_candidatos

    def espiao(indices, gabarito):
        chamadas.append(list(indices))
        return original(indices, gabarito)

    monkeypatch.setattr(enr, "avaliar_candidatos", espiao)
    executar_auto(tmp_path, ctx, max_rodadas=2)
    assert chamadas
    for indices in chamadas:
        assert set(indices) <= ctx["oculto"]


# --------------------------------------------------------------------------
# Proteções e rodadas manuais
# --------------------------------------------------------------------------

def test_verificar_bloqueio_detecta_indice_proibido(ctx):
    proibido = next(iter(ctx["proibidos"]))
    with pytest.raises(ValueError, match="Vazamento"):
        verificar_bloqueio([proibido], ctx["proibidos"], "teste")


def test_base_com_indice_bloqueado_aborta_rodada(ctx, tmp_path):
    dir_rounds = tmp_path / "rounds"
    executar_rodada(1, dir_rounds, ctx)
    arq = dir_rodada(dir_rounds, 1) / "base.csv"
    base = _ler(arq)
    intruso = pd.DataFrame([{"indice": min(ctx["proibidos"]), "rotulo": 0,
                             "origem": "gabarito", "rodada": 1}])
    pd.concat([base, intruso]).to_csv(arq, index=False)
    with pytest.raises(ValueError, match="Vazamento"):
        executar_rodada(2, dir_rounds, ctx)


def test_mesma_rodada_duas_vezes_da_mesma_saida(ctx, tmp_path):
    d1, d2 = tmp_path / "a", tmp_path / "b"
    executar_rodada(1, d1, ctx)
    executar_rodada(1, d2, ctx)
    executar_rodada(1, d2, ctx)  # refaz no mesmo lugar (sem decisões preenchidas)
    for arq in ("base.csv", "candidatos.csv", "revisao_manual.csv", "metricas.json"):
        assert (dir_rodada(d1, 1) / arq).read_bytes() == (dir_rodada(d2, 1) / arq).read_bytes(), arq


def test_revisao_manual_tem_colunas_e_vizinhos(ctx, tmp_path):
    _, rev = _rodada_manual_com_revisao(ctx, tmp_path)
    for col in ("indice", "nome", "genero", "pontuacao", "tipo", "viz1_nome", "viz1_classe",
                "viz3_nome", "viz3_classe", "decisao_humana", "observacao"):
        assert col in rev.columns
    assert (rev["decisao_humana"] == "").all()
    assert set(rev["tipo"]) <= {"alta_confianca", "incerto"}
    assert not set(rev["indice"]) & set(ctx["rotulo_gab"])  # só sem rótulo no gabarito


def test_decisoes_humanas_entram_e_pendentes_ficam_fora(ctx, tmp_path):
    dir_rounds, rev = _rodada_manual_com_revisao(ctx, tmp_path, minimo=3)
    _gravar_decisoes(dir_rounds, rev, {0: "0", 1: "duvida"})  # linha 2 fica vazia
    with pytest.warns(UserWarning, match="pendentes"):
        met = executar_rodada(2, dir_rounds, ctx)
    base = _ler(dir_rodada(dir_rounds, 2) / "base.csv")
    i_neg, i_duv, i_pend = (int(rev.loc[k, "indice"]) for k in (0, 1, 2))
    linha = base[base["indice"] == i_neg].iloc[0]
    assert (linha["rotulo"], linha["origem"], linha["rodada"]) == (0, "humano", 2)
    assert i_duv not in set(base["indice"]) and i_pend not in set(base["indice"])
    cand2 = _ler(dir_rodada(dir_rounds, 2) / "candidatos.csv")
    assert i_duv not in set(cand2["indice"])          # dúvida não volta à revisão
    assert met["excluidos_duvida"] == [i_duv]
    assert met["humanos_neg_incorporados"] == 1 and met["pendentes_rodada_anterior"] == 1


def test_decisao_positiva_vira_positivo_humano(ctx, tmp_path):
    dir_rounds, rev = _rodada_manual_com_revisao(ctx, tmp_path)
    _gravar_decisoes(dir_rounds, rev, {0: " 1 "})
    met = executar_rodada(2, dir_rounds, ctx)
    base = _ler(dir_rodada(dir_rounds, 2) / "base.csv")
    linha = base[base["indice"] == int(rev.loc[0, "indice"])].iloc[0]
    assert (linha["rotulo"], linha["origem"]) == (1, "humano")
    assert met["humanos_pos_incorporados"] == 1 and met["positivos_novos"] >= 1
    assert not met["convergido"]


def test_decisao_invalida_gera_erro_claro(ctx, tmp_path):
    dir_rounds, rev = _rodada_manual_com_revisao(ctx, tmp_path)
    _gravar_decisoes(dir_rounds, rev, {0: "talvez"})
    with pytest.raises(ErroDecisaoHumana, match=r"'talvez' inválida; use 1"):
        executar_rodada(2, dir_rounds, ctx)
    assert not dir_rodada(dir_rounds, 2).exists()


def test_indice_inventado_ou_bloqueado_na_revisao_gera_erro(ctx, tmp_path):
    dir_rounds, rev = _rodada_manual_com_revisao(ctx, tmp_path)
    extra = rev.iloc[[0]].assign(indice=min(ctx["proibidos"]), decisao_humana="1")
    pd.concat([rev, extra]).to_csv(dir_rodada(dir_rounds, 1) / "revisao_manual.csv", index=False)
    with pytest.raises(ErroDecisaoHumana, match="não foi proposto"):
        executar_rodada(2, dir_rounds, ctx)


def test_nao_refaz_rodada_com_decisoes_sem_sobrescrever(ctx, tmp_path):
    dir_rounds, rev = _rodada_manual_com_revisao(ctx, tmp_path)
    _gravar_decisoes(dir_rounds, rev, {0: "0"})
    with pytest.raises(RuntimeError, match="decisão"):
        executar_rodada(1, dir_rounds, ctx)
    executar_rodada(2, dir_rounds, ctx)
    _gravar_decisoes(dir_rounds, rev, {})  # sem decisões, mas a rodada 2 já existe
    with pytest.raises(RuntimeError, match="posteriores"):
        executar_rodada(1, dir_rounds, ctx)


def test_metricas_json_sem_dados_variaveis(ctx, tmp_path):
    executar_rodada(1, tmp_path, ctx)
    met = json.loads((dir_rodada(tmp_path, 1) / "metricas.json").read_text(encoding="utf-8"))
    assert met["rodada"] == 1 and met["modo"] == "manual"
    assert not any("hora" in k or "data" in k or "tempo" in k for k in met)
