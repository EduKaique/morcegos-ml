"""Enriquecimento iterativo da base (Etapa 3).

Cada rodada treina o classificador escolhido na Etapa 2 (``src.config_modelo``)
com a base atual, pontua todas as instâncias fora da base e fora dos gêneros
bloqueados, e seleciona candidatos:

- **alta confiança**: pontuação >= limiar;
- **incertos** (active learning): 0 < pontuação < limiar, os mais próximos do
  limiar primeiro (pontuação 0 não informa nada e fica de fora).

A base só cresce com rótulos **verificados**: candidatos do grupo ``oculto``
são verificados automaticamente pelo gabarito; candidatos sem rótulo vão para
``revisao_manual.csv`` e só entram quando a decisão humana (1, 0 ou duvida)
for preenchida e a rodada seguinte for executada. A predição do modelo sozinha
nunca rotula nada.

Uso, a partir da raiz do repositório:

    python -m src.enriquecimento --rodada 1            # rodada manual (data/rounds/rodada_01/)
    python -m src.enriquecimento --auto --ate-convergir  # simulação (data/rounds/auto/)
"""

import argparse
import json
import shutil
import sys
import time
import unicodedata
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # roda sem janela gráfica

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.carregar_dados import DIR_RAIZ, carregar_tudo
from src.config_modelo import LIMIAR_PADRAO, PARAMS, VARIANTE, construir_modelo
from src.modelos import pontuar
from src.supervisao import _tabela_md, instancias_bloqueadas
from src.validacao import avaliar_candidatos

RANDOM_STATE = 42
DIR_SEEDS = DIR_RAIZ / "data" / "seeds"
DIR_ROUNDS = DIR_RAIZ / "data" / "rounds"
DIR_RESULTADOS = DIR_RAIZ / "results"
SUBDIR_AUTO = "auto"

MAX_INCERTOS = 30
MAX_REVISAO = 100
MAX_RODADAS = 10
N_VIZINHOS = 3

ORIGEM_SEMENTE = "semente"
ORIGEM_GABARITO = "gabarito"
ORIGEM_HUMANO = "humano"
ORIGENS_VERIFICADAS = (ORIGEM_GABARITO, ORIGEM_HUMANO)

TIPO_ALTA = "alta_confianca"
TIPO_INCERTO = "incerto"

DECISOES_VALIDAS = {"1": 1, "0": 0, "duvida": "duvida"}

COLUNAS_BASE = ["indice", "rotulo", "origem", "rodada"]
COLUNAS_CANDIDATOS = ["indice", "nome", "genero", "pontuacao", "tipo", "grupo_gabarito",
                      "rotulo_gabarito", "verificacao"]
COLUNAS_REVISAO = (["indice", "nome", "genero", "pontuacao", "tipo"]
                   + [f"viz{j}_{c}" for j in range(1, N_VIZINHOS + 1) for c in ("nome", "classe")]
                   + ["decisao_humana", "observacao"])


class ErroDecisaoHumana(ValueError):
    """Valor ou índice inválido em ``revisao_manual.csv``."""


# --------------------------------------------------------------------------
# Contexto e caminhos
# --------------------------------------------------------------------------

def _ler(caminho, **kw):
    return pd.read_csv(caminho, keep_default_na=False, **kw)


def carregar_contexto(dir_seeds=DIR_SEEDS):
    """Dados fixos de todas as rodadas: vetores, semente, gabarito e bloqueios."""
    dir_seeds = Path(dir_seeds)
    df, X = carregar_tudo()
    semente = _ler(dir_seeds / "semente_reduzida.csv")
    gabarito = _ler(dir_seeds / "gabarito.csv")
    gabarito["rotulo_gabarito"] = gabarito["rotulo_gabarito"].astype(int)
    teste = set(_ler(dir_seeds / "teste_verificado.csv")["indice"].astype(int))
    bloqueados = set(instancias_bloqueadas(df, dir_seeds / "generos_bloqueados.csv"))
    oculto = gabarito[gabarito["grupo"] == "oculto"]
    return {
        "df": df, "X": X, "semente": semente, "gabarito": gabarito,
        "proibidos": bloqueados | teste,
        "oculto": set(oculto["indice"].astype(int)),
        "grupo": dict(zip(gabarito["indice"].astype(int), gabarito["grupo"])),
        "rotulo_gab": dict(zip(gabarito["indice"].astype(int), gabarito["rotulo_gabarito"])),
        "morcegos_oculto": oculto[oculto["rotulo_gabarito"] == 1],
        "familias_semente": set(semente.loc[semente["rotulo"].astype(int) == 1, "familia"]),
    }


def dir_rodada(dir_modo, n):
    return Path(dir_modo) / f"rodada_{n:02d}"


def rodadas_existentes(dir_modo):
    dir_modo = Path(dir_modo)
    if not dir_modo.exists():
        return []
    return sorted(int(p.name.split("_")[1]) for p in dir_modo.glob("rodada_[0-9][0-9]")
                  if (p / "metricas.json").exists())


# --------------------------------------------------------------------------
# Proteções
# --------------------------------------------------------------------------

def verificar_bloqueio(indices, proibidos, onde):
    """``ValueError`` se algum índice do teste ou de gênero bloqueado aparecer em ``onde``."""
    vazados = sorted(set(int(i) for i in indices) & set(proibidos))
    if vazados:
        raise ValueError(f"Vazamento: {len(vazados)} índice(s) do teste/gêneros bloqueados em "
                         f"{onde}: {vazados[:10]}")


def _decisoes_preenchidas(caminho):
    if not Path(caminho).exists():
        return 0
    rev = _ler(caminho, dtype=str)
    return int((rev["decisao_humana"].str.strip() != "").sum())


def verificar_pode_escrever(dir_modo, n, sobrescrever):
    """Impede apagar decisões humanas ou deixar rodadas posteriores inconsistentes."""
    if sobrescrever:
        return
    posteriores = [r for r in rodadas_existentes(dir_modo) if r > n]
    if posteriores:
        raise RuntimeError(f"Já existem rodadas posteriores à {n} em {dir_modo} ({posteriores}). "
                           "Use --sobrescrever para refazer.")
    preenchidas = _decisoes_preenchidas(dir_rodada(dir_modo, n) / "revisao_manual.csv")
    if preenchidas:
        raise RuntimeError(f"A rodada {n} já tem {preenchidas} decisão(ões) humana(s) preenchida(s) "
                           "em revisao_manual.csv; refazê-la apagaria essas decisões. "
                           "Use --sobrescrever se for mesmo o caso.")


# --------------------------------------------------------------------------
# Base de entrada e decisões humanas
# --------------------------------------------------------------------------

def base_da_semente(semente):
    base = pd.DataFrame({"indice": semente["indice"].astype(int),
                         "rotulo": semente["rotulo"].astype(int)})
    base["origem"] = ORIGEM_SEMENTE
    base["rodada"] = 0
    return base.sort_values("indice").reset_index(drop=True)[COLUNAS_BASE]


def _normalizar(valor):
    v = unicodedata.normalize("NFKD", str(valor)).encode("ascii", "ignore").decode()
    return v.strip().lower()


def ler_decisoes(dir_anterior, base, proibidos):
    """Lê ``revisao_manual.csv`` da rodada anterior.

    Devolve ``(aceitas, duvidas, pendentes)``: ``aceitas`` é um DataFrame
    ``indice, rotulo``; ``duvidas`` e ``pendentes`` são listas de índices.
    Levanta ``ErroDecisaoHumana`` para valor fora de {1, 0, duvida, vazio}, índice
    que não estava na revisão gerada, índice bloqueado ou já na base.
    """
    arq = Path(dir_anterior) / "revisao_manual.csv"
    vazio = pd.DataFrame(columns=["indice", "rotulo"])
    if not arq.exists():
        return vazio, [], []
    rev = _ler(arq, dtype=str)
    gerados = set(_ler(Path(dir_anterior) / "candidatos.csv")
                  .query("verificacao == 'revisao_manual'")["indice"].astype(int))
    na_base = set(base["indice"])
    aceitas, duvidas, pendentes, erros = [], [], [], []
    for pos, linha in rev.iterrows():
        n_linha = pos + 2  # cabeçalho é a linha 1
        try:
            indice = int(linha["indice"])
        except ValueError:
            erros.append(f"linha {n_linha}: indice inválido {linha['indice']!r}")
            continue
        if indice not in gerados:
            erros.append(f"linha {n_linha}: indice {indice} não foi proposto para revisão nesta rodada")
        elif indice in proibidos:
            erros.append(f"linha {n_linha}: indice {indice} é do teste/gênero bloqueado")
        elif indice in na_base:
            erros.append(f"linha {n_linha}: indice {indice} já está na base")
        valor = _normalizar(linha["decisao_humana"])
        if valor == "":
            pendentes.append(indice)
        elif valor not in DECISOES_VALIDAS:
            erros.append(f"linha {n_linha} (indice {indice}): decisao_humana {linha['decisao_humana']!r} "
                         "inválida; use 1 (morcego), 0 (não é morcego), duvida, ou deixe vazio")
        elif DECISOES_VALIDAS[valor] == "duvida":
            duvidas.append(indice)
        else:
            aceitas.append({"indice": indice, "rotulo": DECISOES_VALIDAS[valor]})
    if erros:
        raise ErroDecisaoHumana(f"Erros em {arq}:\n  " + "\n  ".join(erros))
    return (pd.DataFrame(aceitas, columns=["indice", "rotulo"]) if aceitas else vazio,
            duvidas, pendentes)


def base_de_entrada(n, dir_modo, ctx, modo):
    """Base usada no treino da rodada ``n`` e as decisões humanas incorporadas nela."""
    info = {"humanos_pos": 0, "humanos_neg": 0, "duvidas_novas": 0, "pendentes_anteriores": 0,
            "excluidos_duvida": []}
    if n == 1:
        return base_da_semente(ctx["semente"]), info
    anterior = dir_rodada(dir_modo, n - 1)
    if not (anterior / "base.csv").exists():
        raise FileNotFoundError(f"Rodada {n - 1} não encontrada em {anterior}; rode-a antes.")
    base = _ler(anterior / "base.csv")[COLUNAS_BASE]
    met_anterior = json.loads((anterior / "metricas.json").read_text(encoding="utf-8"))
    excluidos = list(met_anterior.get("excluidos_duvida", []))
    if modo == "manual":
        aceitas, duvidas, pendentes = ler_decisoes(anterior, base, ctx["proibidos"])
        if pendentes:
            warnings.warn(f"{len(pendentes)} linha(s) de {anterior / 'revisao_manual.csv'} sem "
                          "decisao_humana: continuam pendentes (fora da base).")
        if len(aceitas):
            novas = aceitas.assign(origem=ORIGEM_HUMANO, rodada=n)[COLUNAS_BASE]
            base = pd.concat([base, novas], ignore_index=True)
        excluidos = sorted(set(excluidos) | set(duvidas))
        info.update(humanos_pos=int((aceitas["rotulo"] == 1).sum()),
                    humanos_neg=int((aceitas["rotulo"] == 0).sum()),
                    duvidas_novas=len(duvidas), pendentes_anteriores=len(pendentes))
    info["excluidos_duvida"] = excluidos
    return base, info


# --------------------------------------------------------------------------
# Rodada
# --------------------------------------------------------------------------

def selecionar_candidatos(indices, s, limiar, max_incertos):
    """``(alta, incertos)`` como DataFrames ``indice, pontuacao, tipo``."""
    tab = pd.DataFrame({"indice": indices, "pontuacao": s})
    alta = (tab[tab["pontuacao"] >= limiar]
            .sort_values(["pontuacao", "indice"], ascending=[False, True]).assign(tipo=TIPO_ALTA))
    inc = tab[(tab["pontuacao"] > 0) & (tab["pontuacao"] < limiar)]
    inc = (inc.assign(_dist=limiar - inc["pontuacao"]).sort_values(["_dist", "indice"])
           .head(max_incertos).drop(columns="_dist").assign(tipo=TIPO_INCERTO))
    return alta, inc


def _vizinhos(ctx, base, indices):
    """``N_VIZINHOS`` vizinhos mais próximos na base (mesma métrica do modelo, sem escala)."""
    if not len(indices):
        return []
    X = ctx["X"]
    nn = NearestNeighbors(n_neighbors=min(N_VIZINHOS, len(base)), metric=PARAMS["metric"],
                          algorithm="brute").fit(X[base["indice"]])
    _, pos = nn.kneighbors(X[np.asarray(indices)])
    nomes = ctx["df"]["nome"].to_numpy()
    saida = []
    for linha in pos:
        viz = {}
        for j, p in enumerate(linha, start=1):
            viz[f"viz{j}_nome"] = nomes[base["indice"].iloc[p]]
            viz[f"viz{j}_classe"] = int(base["rotulo"].iloc[p])
        saida.append(viz)
    return saida


def _recall_oculto(ctx, base):
    """Recall dos morcegos do oculto que estão na base, por família e por gênero."""
    mo = ctx["morcegos_oculto"]
    na_base = set(base.loc[base["rotulo"] == 1, "indice"])
    mo = mo.assign(achado=mo["indice"].astype(int).isin(na_base))
    por_familia = {f: {"n": len(g), "encontrados": int(g["achado"].sum()),
                       "na_semente": f in ctx["familias_semente"]}
                   for f, g in mo.groupby("familia", sort=True)}
    por_genero = {gen: {"familia": g["familia"].iloc[0], "n": len(g),
                        "encontrados": int(g["achado"].sum())}
                  for gen, g in mo.groupby("genero_atual", sort=True)}
    return por_familia, por_genero


def executar_rodada(n, dir_modo, ctx, modo="manual", limiar=LIMIAR_PADRAO,
                    max_incertos=MAX_INCERTOS, max_revisao=MAX_REVISAO, sobrescrever=False,
                    max_rodadas=None):
    """Executa a rodada ``n`` e grava ``base.csv``, ``candidatos.csv``,
    ``revisao_manual.csv`` e ``metricas.json`` em ``dir_modo/rodada_NN``."""
    if modo not in ("manual", "auto"):
        raise ValueError(f"modo inválido: {modo}")
    if n < 1:
        raise ValueError("A rodada 0 é a semente; as rodadas começam em 1.")
    verificar_pode_escrever(dir_modo, n, sobrescrever)

    base, info = base_de_entrada(n, dir_modo, ctx, modo)
    verificar_bloqueio(base["indice"], ctx["proibidos"], f"base de entrada da rodada {n}")
    if base["indice"].duplicated().any():
        raise ValueError(f"Índices repetidos na base da rodada {n}.")
    extras = base[(base["rodada"] > 0) & ~base["origem"].isin(ORIGENS_VERIFICADAS)]
    if len(extras):
        raise ValueError(f"Base com origem não verificada: {extras['indice'].tolist()[:10]}")

    X, df = ctx["X"], ctx["df"]
    fora = set(base["indice"]) | ctx["proibidos"] | set(info["excluidos_duvida"])
    pool = np.array([i for i in df["indice"] if i not in fora], dtype=int)
    modelo = construir_modelo().fit(X[base["indice"]], base["rotulo"].to_numpy())
    s = pontuar(modelo, X[pool])
    alta, inc = selecionar_candidatos(pool, s, limiar, max_incertos)
    cand = pd.concat([alta, inc], ignore_index=True)
    verificar_bloqueio(cand["indice"], ctx["proibidos"], f"candidatos da rodada {n}")

    cand["nome"] = df["nome"].to_numpy()[cand["indice"]]
    cand["genero"] = df["genero"].to_numpy()[cand["indice"]]
    cand["grupo_gabarito"] = cand["indice"].map(ctx["grupo"]).fillna("")
    invalidos = cand[cand["grupo_gabarito"].isin(["semente", "teste"])]
    if len(invalidos):
        raise RuntimeError(f"Candidatos de grupo {set(invalidos['grupo_gabarito'])} (não deveria "
                           f"ocorrer): {invalidos['indice'].tolist()[:10]}")
    cand["rotulo_gabarito"] = cand["indice"].map(ctx["rotulo_gab"]).astype("Int64")
    do_oculto = cand["grupo_gabarito"] == "oculto"
    sem_rotulo = cand[~do_oculto].sort_values(["pontuacao", "indice"], ascending=[False, True])
    no_csv = set(sem_rotulo["indice"].head(max_revisao))
    cand["verificacao"] = np.where(do_oculto, "gabarito",
                                   np.where(cand["indice"].isin(no_csv), "revisao_manual", "fora_do_csv"))

    # Verificação automática pelo gabarito: só o grupo oculto.
    verificados = cand[do_oculto]
    novas = pd.DataFrame({"indice": verificados["indice"].astype(int),
                          "rotulo": verificados["rotulo_gabarito"].astype(int),
                          "origem": ORIGEM_GABARITO, "rodada": n})
    base_saida = pd.concat([base, novas], ignore_index=True)[COLUNAS_BASE]
    verificar_bloqueio(base_saida["indice"], ctx["proibidos"], f"base de saída da rodada {n}")

    revisao = sem_rotulo[sem_rotulo["indice"].isin(no_csv)].reset_index(drop=True)
    viz = pd.DataFrame(_vizinhos(ctx, base, revisao["indice"].to_numpy()), index=revisao.index)
    revisao = pd.concat([revisao, viz], axis=1).assign(decisao_humana="", observacao="")
    revisao = revisao.reindex(columns=COLUNAS_REVISAO)
    revisao["pontuacao"] = revisao["pontuacao"].round(6)
    cand["pontuacao"] = cand["pontuacao"].round(6)

    # Métricas
    oculto_acum = base_saida.loc[base_saida["origem"] == ORIGEM_GABARITO, "indice"].astype(int).tolist()
    nao_oculto = [i for i in oculto_acum if i not in ctx["oculto"]]
    if nao_oculto:
        raise RuntimeError(f"Índices fora do oculto com origem gabarito: {nao_oculto[:10]}")
    aval = avaliar_candidatos(oculto_acum, ctx["gabarito"]) if oculto_acum else None
    alta_oc = verificados[verificados["tipo"] == TIPO_ALTA]
    positivos_gab = int((verificados["rotulo_gabarito"] == 1).sum())
    positivos_novos = positivos_gab + info["humanos_pos"]
    pend_csv, pend_fora = len(revisao), int((cand["verificacao"] == "fora_do_csv").sum())
    pendentes = pend_csv + pend_fora
    total_oculto = len(ctx["morcegos_oculto"])
    por_familia, por_genero = _recall_oculto(ctx, base_saida)

    convergido, motivo = False, ""
    if positivos_novos == 0 and (modo == "auto" or pendentes == 0):
        convergido = True
        motivo = "nenhum positivo novo confirmado"
        motivo += (f"; {pendentes} candidato(s) sem rótulo ficaram pendentes (não resolvíveis no "
                   "modo automático)" if modo == "auto" else " e nenhuma pendência de revisão")
    elif max_rodadas is not None and n >= max_rodadas:
        motivo = f"limite de {max_rodadas} rodadas atingido sem convergir"
    elif positivos_novos == 0:
        motivo = f"nenhum positivo novo, mas {pendentes} pendência(s) de revisão manual"

    metricas = {
        "rodada": n, "modo": modo, "limiar": limiar, "modelo": VARIANTE,
        "base_pos_entrada": int((base["rotulo"] == 1).sum()),
        "base_neg_entrada": int((base["rotulo"] == 0).sum()),
        "base_pos_saida": int((base_saida["rotulo"] == 1).sum()),
        "base_neg_saida": int((base_saida["rotulo"] == 0).sum()),
        "humanos_pos_incorporados": info["humanos_pos"],
        "humanos_neg_incorporados": info["humanos_neg"],
        "duvidas_novas": info["duvidas_novas"],
        "pendentes_rodada_anterior": info["pendentes_anteriores"],
        "n_pool": len(pool),
        "candidatos_alta": len(alta), "candidatos_incertos": len(inc),
        "alta_do_oculto": len(alta_oc), "alta_sem_rotulo": int((~do_oculto & (cand["tipo"] == TIPO_ALTA)).sum()),
        "incertos_do_oculto": int((do_oculto & (cand["tipo"] == TIPO_INCERTO)).sum()),
        "incertos_sem_rotulo": int((~do_oculto & (cand["tipo"] == TIPO_INCERTO)).sum()),
        "positivos_novos_gabarito": positivos_gab,
        "positivos_novos": positivos_novos,
        "negativos_dificeis": int((verificados["rotulo_gabarito"] == 0).sum()),
        "falsos_positivos_alta": int((alta_oc["rotulo_gabarito"] == 0).sum()),
        "precisao_alta_verificada": (float((alta_oc["rotulo_gabarito"] == 1).mean())
                                     if len(alta_oc) else None),
        "pendentes_revisao_csv": pend_csv, "pendentes_fora_do_csv": pend_fora,
        "pendentes_total": pendentes,
        "morcegos_oculto_total": total_oculto,
        "morcegos_oculto_encontrados_acum": int(aval["morcegos_reais_acum"].iloc[-1]) if aval is not None else 0,
        "recall_oculto_acum": float(aval["recall_acum"].iloc[-1]) if aval is not None else 0.0,
        "precisao_verificados_acum": float(aval["precisao_acum"].iloc[-1]) if aval is not None else None,
        "convergido": convergido, "motivo": motivo,
        "excluidos_duvida": info["excluidos_duvida"],
        "recall_por_familia": por_familia, "recall_por_genero": por_genero,
    }

    destino = dir_rodada(dir_modo, n)
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)
    base_saida.to_csv(destino / "base.csv", index=False)
    cand[COLUNAS_CANDIDATOS].to_csv(destino / "candidatos.csv", index=False)
    revisao.to_csv(destino / "revisao_manual.csv", index=False)
    (destino / "metricas.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return metricas


def executar_auto(dir_rounds=DIR_ROUNDS, ctx=None, limiar=LIMIAR_PADRAO, max_incertos=MAX_INCERTOS,
                  max_revisao=MAX_REVISAO, max_rodadas=MAX_RODADAS, dir_seeds=DIR_SEEDS):
    """Simulação: rodadas com verificação só pelo gabarito, até convergir ou ``max_rodadas``.

    Recomeça do zero em ``dir_rounds/auto`` (separado das rodadas manuais).
    """
    ctx = ctx or carregar_contexto(dir_seeds)
    dir_auto = Path(dir_rounds) / SUBDIR_AUTO
    if dir_auto.exists():
        shutil.rmtree(dir_auto)
    historico = []
    for n in range(1, max_rodadas + 1):
        met = executar_rodada(n, dir_auto, ctx, "auto", limiar, max_incertos, max_revisao,
                              max_rodadas=max_rodadas)
        historico.append(met)
        if met["convergido"]:
            break
    return historico


# --------------------------------------------------------------------------
# Saídas agregadas: histórico, figuras e relatório
# --------------------------------------------------------------------------

def ler_metricas(dir_modo):
    return [json.loads((dir_rodada(dir_modo, n) / "metricas.json").read_text(encoding="utf-8"))
            for n in rodadas_existentes(dir_modo)]


def montar_historico(dir_rounds):
    linhas = []
    for dir_modo in (Path(dir_rounds) / SUBDIR_AUTO, Path(dir_rounds)):
        for met in ler_metricas(dir_modo):
            linhas.append({k: v for k, v in met.items() if not isinstance(v, (dict, list))})
    return pd.DataFrame(linhas)


def _pendentes_com_vizinhos(dir_r):
    cand = _ler(dir_r / "candidatos.csv")
    pend = cand[cand["verificacao"] != "gabarito"]
    rev = _ler(dir_r / "revisao_manual.csv")
    cols = [c for c in rev.columns if c.startswith("viz")]
    return (pend.merge(rev[["indice"] + cols], on="indice", how="left")
            .sort_values(["pontuacao", "indice"], ascending=[False, True]))


def figura_positivos(hist, semente_pos, caminho):
    fig, eixo = plt.subplots(figsize=(7, 4.5))
    for modo, g in hist.groupby("modo"):
        x = [0] + g["rodada"].tolist()
        eixo.plot(x, [semente_pos] + g["base_pos_saida"].tolist(), marker="o",
                  label=f"modo {modo}")
    eixo.set_xlabel("rodada")
    eixo.set_ylabel("positivos na base (acumulado)")
    eixo.set_title("Positivos verificados por rodada (rodada 0 = semente)")
    eixo.legend()
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def figura_recall_familia(metricas, caminho):
    fig, eixo = plt.subplots(figsize=(8, 4.5))
    if metricas:
        familias = sorted(metricas[0]["recall_por_familia"])
        x = [0] + [m["rodada"] for m in metricas]
        for f in familias:
            info = metricas[0]["recall_por_familia"][f]
            y = [0] + [m["recall_por_familia"][f]["encontrados"] / info["n"] for m in metricas]
            estrela = "" if info["na_semente"] else "*"
            eixo.plot(x, y, marker="o", label=f"{f}{estrela} (n={info['n']})")
        eixo.legend(fontsize=7, loc="lower right")
    eixo.set_ylim(-0.05, 1.05)
    eixo.set_xlabel("rodada")
    eixo.set_ylabel("recall acumulado no oculto")
    eixo.set_title("Recall do oculto por família (* = sem representante na semente)")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def figura_umap(ctx, dir_r, Y, caminho):
    base = _ler(dir_r / "base.csv")
    cand = _ler(dir_r / "candidatos.csv")
    na_base_pos = set(base.loc[base["rotulo"] == 1, "indice"])
    sem_pos = ctx["semente"].loc[ctx["semente"]["rotulo"].astype(int) == 1, "indice"].astype(int)
    mo = ctx["morcegos_oculto"]["indice"].astype(int)
    grupos = [
        ("morcegos da semente", sem_pos, "tab:blue", "o"),
        ("oculto encontrado", mo[mo.isin(na_base_pos)], "tab:green", "^"),
        ("oculto não encontrado", mo[~mo.isin(na_base_pos)], "tab:red", "x"),
        ("candidatos pendentes (sem rótulo)",
         cand.loc[cand["verificacao"] != "gabarito", "indice"], "tab:orange", "s"),
    ]
    fig, eixo = plt.subplots(figsize=(8, 7))
    eixo.scatter(Y[:, 0], Y[:, 1], s=2, color="lightgray", linewidths=0, label="demais instâncias")
    for rotulo, idx, cor, marca in grupos:
        idx = np.asarray(list(idx), dtype=int)
        eixo.scatter(Y[idx, 0], Y[idx, 1], s=22, color=cor, marker=marca,
                     label=f"{rotulo} ({len(idx)})")
    eixo.set_title(f"Projeção 2D (random_state=42) após a {dir_r.name.replace('_', ' ')}")
    eixo.legend(fontsize=8, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def _secao_modo(titulo, metricas, dir_modo, max_revisao):
    if not metricas:
        return [f"## {titulo}", "", "_(nenhuma rodada executada)_", ""]
    ultima = metricas[-1]
    dir_r = dir_rodada(dir_modo, ultima["rodada"])
    cols = {"rodada": "rodada", "base_pos_saida": "positivos", "base_neg_saida": "negativos",
            "candidatos_alta": "alta", "candidatos_incertos": "incertos",
            "positivos_novos": "pos. novos", "negativos_dificeis": "neg. difíceis",
            "falsos_positivos_alta": "FP alta", "precisao_alta_verificada": "precisão alta",
            "morcegos_oculto_encontrados_acum": "oculto encontrados",
            "recall_oculto_acum": "recall oculto", "pendentes_total": "pendentes",
            "convergido": "convergido"}
    tab = pd.DataFrame(metricas)[list(cols)].rename(columns=cols)
    for c in ("precisão alta", "recall oculto"):
        tab[c] = tab[c].map(lambda v: "-" if v is None or pd.isna(v) else f"{v:.3f}")
    linhas = [f"## {titulo}", "", f"Pasta: `{_caminho_relativo(dir_modo)}`. Positivos e negativos na base "
              "ao fim de cada rodada; \"pos. novos\" inclui decisões humanas incorporadas.", "",
              _tabela_md(tab),
              f"**Motivo de parada / estado:** {ultima['motivo'] or 'em andamento'}.", ""]

    linhas += ["### Recuperação do oculto por família", ""]
    fam = pd.DataFrame([{"familia": f + ("" if v["na_semente"] else "*"), "n": v["n"],
                         "encontrados": v["encontrados"], "recall": f"{v['encontrados'] / v['n']:.0%}"}
                        for f, v in ultima["recall_por_familia"].items()])
    linhas += [_tabela_md(fam), "\\* família sem representante na semente.", "",
               "### Recuperação do oculto por gênero", ""]
    gen = pd.DataFrame([{"familia": v["familia"], "genero": g, "n": v["n"],
                         "encontrados": v["encontrados"], "recall": f"{v['encontrados'] / v['n']:.0%}"}
                        for g, v in ultima["recall_por_genero"].items()]).sort_values(["familia", "genero"])
    linhas += [_tabela_md(gen), "### Pendências (candidatos sem rótulo, última rodada)", ""]
    if ultima["pendentes_fora_do_csv"]:
        linhas += [f"> **Aviso:** {ultima['pendentes_total']} candidatos sem rótulo, mais que "
                   f"`--max-revisao` ({max_revisao}); só os {ultima['pendentes_revisao_csv']} de "
                   f"maior pontuação foram para `revisao_manual.csv` e "
                   f"{ultima['pendentes_fora_do_csv']} ficaram de fora.", ""]
    pend = _pendentes_com_vizinhos(dir_r)
    linhas += [f"{len(pend)} candidato(s) sem rótulo; os 10 de maior pontuação:", ""]
    if len(pend):
        mostrar = pend.head(10).copy()
        mostrar["vizinhos (classe)"] = [
            "; ".join(f"{r[f'viz{j}_nome']} ({r[f'viz{j}_classe']})" for j in range(1, N_VIZINHOS + 1)
                      if pd.notna(r.get(f"viz{j}_nome")) and r[f"viz{j}_nome"] != "")
            for _, r in mostrar.iterrows()]
        linhas += [_tabela_md(mostrar[["indice", "nome", "pontuacao", "tipo", "vizinhos (classe)"]])]
    else:
        linhas += ["_(nenhum)_", ""]
    if ultima["excluidos_duvida"]:
        linhas += [f"Marcados como dúvida (fora da base e da revisão): {ultima['excluidos_duvida']}", ""]
    return linhas


def _caminho_relativo(caminho):
    """Caminho relativo à raiz do repositório (sem expor o diretório do usuário)."""
    try:
        return Path(caminho).resolve().relative_to(DIR_RAIZ).as_posix()
    except ValueError:
        return Path(caminho).name


def gerar_relatorio(dir_rounds, limiar, max_incertos, max_revisao, tempo_s):
    dir_rounds = Path(dir_rounds)
    auto = ler_metricas(dir_rounds / SUBDIR_AUTO)
    manual = ler_metricas(dir_rounds)
    linhas = [
        "# Relatório da Etapa 3 — enriquecimento iterativo", "",
        "Índices na base 0. Rodada 0 = `data/seeds/semente_reduzida.csv`.", "",
        "## Configuração", "",
        f"- Classificador fixo (`src/config_modelo.py`, decisão da equipe a partir de "
        f"`results/relatorio_etapa2.md`): `{VARIANTE}`, {PARAMS}, sem escala.",
        f"- Limiar de alta confiança: {limiar}. Incertos: até {max_incertos} com 0 < pontuação < "
        "limiar, mais próximos do limiar primeiro (pontuação 0 não informa nada).",
        f"- Revisão manual: até {max_revisao} candidatos sem rótulo por rodada, por pontuação.",
        "- A base só cresce com rótulos verificados: `gabarito` (grupo oculto, verificação "
        "automática) ou `humano` (decisão registrada em `revisao_manual.csv`). Teste e gêneros "
        "bloqueados são verificados antes e depois de cada rodada.",
        "- Convergência: rodada sem positivo novo confirmado e sem pendências (no modo automático, "
        "as pendências sem rótulo não são resolvíveis e são só registradas).",
        "- Precisão alta = morcegos / candidatos de alta confiança verificados pelo gabarito na "
        "rodada. Recall do oculto calculado com `src.validacao.avaliar_candidatos` sobre a lista "
        "acumulada de índices do oculto verificados.", "",
    ]
    linhas += _secao_modo("Modo automático (simulação, só gabarito)", auto,
                          dir_rounds / SUBDIR_AUTO, max_revisao)
    linhas += _secao_modo("Rodadas manuais", manual, dir_rounds, max_revisao)
    linhas += [
        "## Limites", "",
        "- O gabarito faz o papel do especialista no modo automático: é uma simulação otimista "
        "(verificação instantânea e sem erro).",
        "- A pontuação do k-NN com k=3 é discreta; quase todo o pool tem pontuação 0, então há "
        "poucos incertos e o active learning tem pouco material.",
        "- O oculto concentra os morcegos em *Myotis* (28) e *Rhinolophus* (10): o recall agregado "
        "muda aos saltos com esses dois gêneros.",
        "- Os negativos prováveis da semente não foram verificados taxonomicamente.",
        "- Candidatos sem rótulo não entram na base no modo automático; morcegos fora da lista "
        "taxonômica só seriam incorporados por decisão humana.",
        "- O teste congelado não é usado aqui (Etapa 4).", "",
        "## Figuras", "",
        "- `results/figuras/etapa3_positivos_por_rodada.png`",
        "- `results/figuras/etapa3_recall_familia_por_rodada.png`",
        "- `results/figuras/etapa3_umap.png`", "",
        "## Execução", "", f"Tempo do último comando (rodadas, projeção e figuras): {tempo_s:.1f} s.", "",
    ]
    return "\n".join(linhas)


def gerar_saidas(ctx, dir_rounds=DIR_ROUNDS, dir_resultados=DIR_RESULTADOS, limiar=LIMIAR_PADRAO,
                 max_incertos=MAX_INCERTOS, max_revisao=MAX_REVISAO, projetar=None, inicio=None):
    """Regera histórico, figuras e relatório a partir de todas as rodadas gravadas."""
    dir_rounds, dir_resultados = Path(dir_rounds), Path(dir_resultados)
    dir_figuras = dir_resultados / "figuras"
    dir_figuras.mkdir(parents=True, exist_ok=True)
    hist = montar_historico(dir_rounds)
    hist.to_csv(dir_resultados / "etapa3_historico.csv", index=False)

    semente_pos = int((ctx["semente"]["rotulo"].astype(int) == 1).sum())
    if len(hist):
        figura_positivos(hist, semente_pos, dir_figuras / "etapa3_positivos_por_rodada.png")
    auto = ler_metricas(dir_rounds / SUBDIR_AUTO)
    principal = auto or ler_metricas(dir_rounds)
    dir_principal = dir_rounds / SUBDIR_AUTO if auto else dir_rounds
    figura_recall_familia(principal, dir_figuras / "etapa3_recall_familia_por_rodada.png")
    if principal:
        if projetar is None:
            from src.exploracao import projetar_umap as projetar
        Y = projetar(ctx["X"])
        figura_umap(ctx, dir_rodada(dir_principal, principal[-1]["rodada"]), Y,
                    dir_figuras / "etapa3_umap.png")
    (dir_resultados / "relatorio_etapa3.md").write_text(
        gerar_relatorio(dir_rounds, limiar, max_incertos, max_revisao,
                        time.perf_counter() - inicio if inicio is not None else 0.0),
        encoding="utf-8")
    return hist


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Enriquecimento iterativo da base (Etapa 3).")
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--rodada", type=int, help="executa a rodada manual N (data/rounds/rodada_NN)")
    grupo.add_argument("--auto", action="store_true",
                       help="simulação só com o gabarito, em data/rounds/auto (exige --ate-convergir)")
    parser.add_argument("--ate-convergir", action="store_true",
                        help="no modo automático, roda até convergir ou --max-rodadas")
    parser.add_argument("--max-rodadas", type=int, default=MAX_RODADAS)
    parser.add_argument("--limiar", type=float, default=LIMIAR_PADRAO,
                        help=f"limiar de alta confiança (padrão: {LIMIAR_PADRAO}, da Etapa 2)")
    parser.add_argument("--max-incertos", type=int, default=MAX_INCERTOS)
    parser.add_argument("--max-revisao", type=int, default=MAX_REVISAO)
    parser.add_argument("--sobrescrever", action="store_true",
                        help="permite refazer uma rodada com decisões preenchidas ou com rodadas posteriores")
    args = parser.parse_args()
    if args.auto and not args.ate_convergir:
        parser.error("use --auto --ate-convergir")

    inicio = time.perf_counter()
    ctx = carregar_contexto()
    try:
        if args.auto:
            hist = executar_auto(ctx=ctx, limiar=args.limiar, max_incertos=args.max_incertos,
                                 max_revisao=args.max_revisao, max_rodadas=args.max_rodadas)
            ultima = hist[-1]
            print(f"Modo automático: {len(hist)} rodada(s) em {DIR_ROUNDS / SUBDIR_AUTO}")
        else:
            ultima = executar_rodada(args.rodada, DIR_ROUNDS, ctx, "manual", args.limiar,
                                     args.max_incertos, args.max_revisao, args.sobrescrever)
            print(f"Rodada {args.rodada} gravada em {dir_rodada(DIR_ROUNDS, args.rodada)}")
            print(f"Revisão manual: {ultima['pendentes_revisao_csv']} linha(s) em revisao_manual.csv")
    except (ValueError, RuntimeError, FileNotFoundError) as erro:
        sys.exit(f"ERRO: {erro}")
    print(f"Morcegos do oculto encontrados: {ultima['morcegos_oculto_encontrados_acum']}/"
          f"{ultima['morcegos_oculto_total']}; pendentes sem rótulo: {ultima['pendentes_total']}")
    print(f"Estado: {ultima['motivo'] or 'em andamento'}")
    gerar_saidas(ctx, limiar=args.limiar, max_incertos=args.max_incertos,
                 max_revisao=args.max_revisao, inicio=inicio)
    print(f"Relatório em {DIR_RESULTADOS / 'relatorio_etapa3.md'} "
          f"(tempo: {time.perf_counter() - inicio:.1f} s)")


if __name__ == "__main__":
    main()
