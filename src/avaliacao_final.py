"""Modelo final e testes de performance (Etapa 4).

Treina o classificador fixo (``src.config_modelo``) com a base final da última
rodada manual convergida (``data/rounds/rodada_NN/base.csv``) e mede o
desempenho de forma honesta:

- **teste congelado** (``data/seeds/teste_verificado.csv``): só é lido com
  ``--avaliar-teste``, depois de conferir o SHA256; cada uso é registrado em
  ``results/etapa4_log_uso_teste.csv``. Métricas no limiar fixo com intervalos
  de Wilson (proporções) e bootstrap estratificado (F1 e AP);
- **conjunto supervisionado final** (base final ∪ teste): validação agrupada por
  gênero repetida (sementes 42..51) e *leave-one-genus-out* dos gêneros de morcego.

Nenhum hiperparâmetro nem o limiar é ajustado aqui: tudo vem de
``src.config_modelo`` (decidido na Etapa 2). Sem ``--avaliar-teste`` o script só
valida a rodada, confere vazamento, treina o modelo final e mostra um resumo;
nada é gravado em ``results/``.

Uso, a partir da raiz do repositório:

    python -m src.avaliacao_final                                 # só preparação
    python -m src.avaliacao_final --rodada-final 3 --avaliar-teste
"""

import argparse
import csv
import io
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # roda sem janela gráfica

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors

from src.carregar_dados import DIR_RAIZ, carregar_tudo
from src.config_modelo import (ESCALA, LIMIAR_PADRAO, PARAMS, RANDOM_STATE, VARIANTE,
                               construir_modelo)
from src.enriquecimento import COLUNAS_BASE, DIR_ROUNDS, dir_rodada, rodadas_existentes, verificar_bloqueio
from src.modelos import pontuar
from src.supervisao import (_tabela_md, carregar_sinonimos, carregar_taxonomia,
                            instancias_bloqueadas, sha256_bytes)
from src.validacao import validacao_agrupada

DIR_SEEDS = DIR_RAIZ / "data" / "seeds"
DIR_RESULTADOS = DIR_RAIZ / "results"

SEMENTES_REPETICAO = tuple(range(42, 52))
N_SPLITS = 5
N_BOOTSTRAP = 2000
N_VIZINHOS = 3
N_QUASE = 10
Z_95 = 1.959963984540054

ARQ_LOG = "etapa4_log_uso_teste.csv"
COLUNAS_LOG = ["data_hora", "hash_teste", "configuracao", "limiar", "rodada_final"]
ARQUIVOS_SAIDA = ("etapa4_metricas_teste.csv", "etapa4_matriz_confusao_teste.csv",
                  "etapa4_cv_conjunto_final.csv", "etapa4_logo_generos.csv", "etapa4_erros.csv",
                  "etapa4_pontuacoes_todas.csv", "relatorio_etapa4.md")
FIGURAS_SAIDA = ("etapa4_matriz_confusao_teste.png", "etapa4_pontuacoes_teste.png",
                 "etapa4_ic_wilson_teste.png", "etapa4_logo_generos.png")

CONFIGURACAO = f"{VARIANTE}|" + "|".join(f"{k}={v}" for k, v in PARAMS.items()) + \
    ("|escala" if ESCALA else "|sem_escala")


def _ler(caminho, **kw):
    return pd.read_csv(caminho, keep_default_na=False, **kw)


def _csv(tabela, caminho):
    tabela.to_csv(caminho, index=False, lineterminator="\n")


# --------------------------------------------------------------------------
# Entradas
# --------------------------------------------------------------------------

def carregar_base_final(dir_rounds=DIR_ROUNDS, n=None):
    """Base da rodada manual ``n`` (padrão: a mais recente), exigindo convergência.

    Devolve ``(base, n, metricas)``. Levanta ``FileNotFoundError`` se não houver
    a rodada e ``RuntimeError`` se ela não convergiu.
    """
    dir_rounds = Path(dir_rounds)
    existentes = rodadas_existentes(dir_rounds)
    if not existentes:
        raise FileNotFoundError(f"Nenhuma rodada manual em {dir_rounds}; rode a Etapa 3 antes.")
    if n is None:
        n = existentes[-1]
    elif n not in existentes:
        raise FileNotFoundError(f"Rodada {n} não encontrada em {dir_rounds} (existem {existentes}).")
    elif n < existentes[-1]:
        warnings.warn(f"Rodada {n} escolhida, mas existem rodadas posteriores: "
                      f"{[r for r in existentes if r > n]}.")
    dir_r = dir_rodada(dir_rounds, n)
    met = json.loads((dir_r / "metricas.json").read_text(encoding="utf-8"))
    if met.get("convergido") is not True:
        raise RuntimeError(f"A rodada {n} não convergiu (convergido={met.get('convergido')}); "
                           "o modelo final exige a base de uma rodada convergida.")
    base = _ler(dir_r / "base.csv")[COLUNAS_BASE]
    base["indice"] = base["indice"].astype(int)
    base["rotulo"] = base["rotulo"].astype(int)
    if base["indice"].duplicated().any():
        raise ValueError(f"Índices repetidos em {dir_r / 'base.csv'}.")
    return base.sort_values("indice").reset_index(drop=True), n, met


def carregar_sementes(dir_seeds=DIR_SEEDS):
    """Semente, gabarito (sem o grupo ``teste``) e bloqueios. Não lê o teste."""
    dir_seeds = Path(dir_seeds)
    semente = _ler(dir_seeds / "semente_reduzida.csv")
    gabarito = _ler(dir_seeds / "gabarito.csv")
    gabarito = gabarito[gabarito["grupo"] != "teste"].reset_index(drop=True)
    arq_bloq = dir_seeds / "generos_bloqueados.csv"
    return {"semente": semente, "gabarito": gabarito, "arq_bloqueados": arq_bloq,
            "generos_bloqueados": set(_ler(arq_bloq)["genero_atual"])}


def ler_teste_verificado(dir_seeds=DIR_SEEDS):
    """Única porta de leitura do teste congelado: confere o SHA256 antes de devolver.

    Devolve ``(teste, hash)``; levanta ``RuntimeError`` se o hash não bater.
    """
    dir_seeds = Path(dir_seeds)
    conteudo = (dir_seeds / "teste_verificado.csv").read_bytes()
    esperado = (dir_seeds / "teste_verificado.sha256").read_text(encoding="utf-8").split()[0]
    obtido = sha256_bytes(conteudo)
    if obtido != esperado:
        raise RuntimeError(f"SHA256 do teste congelado não confere (esperado {esperado[:12]}…, "
                           f"obtido {obtido[:12]}…). O teste foi alterado; avaliação abortada.")
    teste = pd.read_csv(io.BytesIO(conteudo), keep_default_na=False)
    teste["indice"] = teste["indice"].astype(int)
    teste["rotulo"] = teste["rotulo"].astype(int)
    return teste, obtido


def anotar(df, fontes):
    """``df`` com ``genero_atual``, ``ordem`` e ``familia`` para todas as instâncias.

    ``genero_atual`` aplica os sinônimos (como ``instancias_bloqueadas``); ordem e
    família vêm da lista taxonômica pelo gênero. As linhas de ``fontes``
    (semente, gabarito, teste) têm prioridade.
    """
    sin = carregar_sinonimos()
    ann = df[["indice", "nome", "genero"]].copy().set_index("indice", drop=False)
    ann["genero_atual"] = ann["genero"].map(lambda g: sin.get(g, g))
    tax = carregar_taxonomia().set_index("genero")
    for col in ("ordem", "familia"):
        ann[col] = ann["genero_atual"].map(tax[col]).fillna("")
    for fonte in fontes:
        f = fonte.set_index(fonte["indice"].astype(int))
        for col in ("genero_atual", "ordem", "familia"):
            valores = f[col][f[col] != ""]
            ann.loc[valores.index, col] = valores
    return ann.reset_index(drop=True)


def verificar_base(base, ann, bloqueados_idx, generos_bloqueados, teste_idx=None):
    """``ValueError`` se a base tiver índice do teste/bloqueado ou gênero bloqueado."""
    proibidos = set(bloqueados_idx) | set(teste_idx if teste_idx is not None else ())
    verificar_bloqueio(base["indice"], proibidos, "base de treino do modelo final")
    generos = set(ann.loc[base["indice"], "genero_atual"]) & set(generos_bloqueados)
    if generos:
        raise ValueError(f"Vazamento: gêneros bloqueados na base final: {sorted(generos)[:10]}")
    if teste_idx is not None:
        fora = set(teste_idx) - set(bloqueados_idx)
        if fora:
            raise ValueError(f"Instâncias do teste fora dos gêneros bloqueados: {sorted(fora)[:10]}")


def registrar_uso_teste(arq_log, hash_teste, rodada):
    """Acrescenta uma linha ao log de uso do teste; avisa se já havia uso anterior."""
    arq_log = Path(arq_log)
    anteriores = len(_ler(arq_log)) if arq_log.exists() else 0
    if anteriores:
        warnings.warn(f"O teste congelado já foi avaliado {anteriores} vez(es) ({arq_log.name}). "
                      "Repetir a avaliação enquanto se altera o modelo invalida o teste.")
    arq_log.parent.mkdir(parents=True, exist_ok=True)
    novo = not arq_log.exists()
    with open(arq_log, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        if novo:
            w.writerow(COLUNAS_LOG)
        w.writerow([datetime.now().isoformat(timespec="seconds"), hash_teste, CONFIGURACAO,
                    LIMIAR_PADRAO, rodada])
    return anteriores


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------

def wilson(k, n, z=Z_95):
    """Proporção ``k/n`` e intervalo de Wilson ``(p, inf, sup)``; NaN se ``n == 0``."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    centro = (p + z ** 2 / (2 * n)) / den
    meia = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (p, max(0.0, centro - meia), min(1.0, centro + meia))


def contagens(y, s, limiar=LIMIAR_PADRAO):
    y = np.asarray(y).astype(int)
    pred = np.asarray(s) >= limiar
    return {"vp": int((pred & (y == 1)).sum()), "fp": int((pred & (y == 0)).sum()),
            "vn": int((~pred & (y == 0)).sum()), "fn": int((~pred & (y == 1)).sum())}


def proporcoes(c):
    """``{metrica: (k, n)}`` das quatro proporções binomiais."""
    return {"sensibilidade": (c["vp"], c["vp"] + c["fn"]),
            "especificidade": (c["vn"], c["vn"] + c["fp"]),
            "precisao": (c["vp"], c["vp"] + c["fp"]),
            "acuracia": (c["vp"] + c["vn"], sum(c.values()))}


def f1(c):
    den = 2 * c["vp"] + c["fp"] + c["fn"]
    return 2 * c["vp"] / den if den else np.nan


def metricas_limiar(y, s, limiar=LIMIAR_PADRAO):
    """Contagens, proporções, F1 (no limiar fixo) e AP (sem limiar)."""
    c = contagens(y, s, limiar)
    m = {**c, **{nome: (k / n if n else np.nan) for nome, (k, n) in proporcoes(c).items()}}
    m["f1"] = f1(c)
    m["ap"] = float(average_precision_score(y, s)) if np.any(np.asarray(y) == 1) else np.nan
    return m


def bootstrap_f1_ap(y, s, limiar=LIMIAR_PADRAO, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    """Intervalo percentil 2,5–97,5 de F1 e AP, reamostrando cada classe separadamente."""
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    f1s, aps = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        f1s[b] = f1(contagens(y[idx], s[idx], limiar))
        aps[b] = average_precision_score(y[idx], s[idx])
    return {"f1": tuple(np.nanpercentile(f1s, [2.5, 97.5])),
            "ap": tuple(np.percentile(aps, [2.5, 97.5]))}


def tabela_metricas_teste(y, s, n_boot, limiar=LIMIAR_PADRAO):
    """Tabela ``metrica, valor, ic_inf, ic_sup, metodo, k, n`` do teste."""
    c = contagens(y, s, limiar)
    linhas = []
    for nome, (k, n) in proporcoes(c).items():
        p, inf, sup = wilson(k, n)
        linhas.append({"metrica": nome, "valor": p, "ic_inf": inf, "ic_sup": sup,
                       "metodo": "wilson", "k": k, "n": n})
    boot = bootstrap_f1_ap(y, s, limiar, n_boot)
    linhas.append({"metrica": "f1", "valor": f1(c), "ic_inf": boot["f1"][0], "ic_sup": boot["f1"][1],
                   "metodo": f"bootstrap_estratificado_{n_boot}", "k": 2 * c["vp"],
                   "n": 2 * c["vp"] + c["fp"] + c["fn"]})
    linhas.append({"metrica": "precisao_media_ap", "valor": float(average_precision_score(y, s)),
                   "ic_inf": boot["ap"][0], "ic_sup": boot["ap"][1],
                   "metodo": f"bootstrap_estratificado_{n_boot}", "k": np.nan, "n": len(y)})
    n_neg = int((np.asarray(y) == 0).sum())
    p, inf, sup = wilson(n_neg, len(y))
    linhas.append({"metrica": "acuracia_linha_base_tudo_0", "valor": p, "ic_inf": inf,
                   "ic_sup": sup, "metodo": "wilson", "k": n_neg, "n": len(y)})
    tab = pd.DataFrame(linhas)
    tab["k"] = tab["k"].astype("Int64")
    return tab, c


def matriz_confusao(c):
    return pd.DataFrame({"real": ["morcego (1)", "outro (0)"],
                         "previsto_morcego": [int(c["vp"]), int(c["fp"])],
                         "previsto_outro": [int(c["fn"]), int(c["vn"])]})


def negativos_por_origem(teste, s, limiar=LIMIAR_PADRAO):
    neg = teste.assign(pontuacao=s)[teste["rotulo"] == 0]
    linhas = []
    for origem, g in neg.groupby("origem", sort=True):
        fp = int((g["pontuacao"] >= limiar).sum())
        p, inf, sup = wilson(len(g) - fp, len(g))
        linhas.append({"origem": origem, "verificado": origem != "negativo_provavel",
                       "n": len(g), "fp": fp, "especificidade": p, "ic_inf": inf, "ic_sup": sup})
    return pd.DataFrame(linhas)


def fp_por_ordem(teste, s, limiar=LIMIAR_PADRAO):
    neg = teste.assign(pontuacao=s)[teste["rotulo"] == 0].copy()
    neg["ordem"] = neg["ordem"].replace("", "(sem ordem: negativo provável)")
    tab = (neg.assign(fp=neg["pontuacao"] >= limiar).groupby("ordem")
           .agg(n_negativos=("indice", "size"), fp=("fp", "sum")).reset_index())
    tab["fp"] = tab["fp"].astype(int)
    return tab.sort_values(["fp", "n_negativos", "ordem"], ascending=[False, False, True])


# --------------------------------------------------------------------------
# Conjunto supervisionado final
# --------------------------------------------------------------------------

def conjunto_final(base, teste, ann, semente):
    """Base final ∪ teste com rótulos verificados e a origem de cada rótulo."""
    origem_sem = dict(zip(semente["indice"].astype(int), semente["origem"]))
    b = base[["indice", "rotulo"]].assign(
        conjunto="base",
        origem_rotulo=[origem_sem[i] if o == "semente" else
                       ("gabarito_taxonomia" if o == "gabarito" else o)
                       for i, o in zip(base["indice"], base["origem"])])
    t = teste[["indice", "rotulo", "origem"]].rename(columns={"origem": "origem_rotulo"})
    t = t.assign(conjunto="teste")
    conj = pd.concat([b, t], ignore_index=True)
    if conj["indice"].duplicated().any():
        raise ValueError("Instâncias repetidas entre base final e teste.")
    conj = conj.merge(ann[["indice", "nome", "genero", "genero_atual", "ordem", "familia"]],
                      on="indice", how="left")
    return conj.sort_values("indice").reset_index(drop=True)


def gerar_dobras(conj, seeds=SEMENTES_REPETICAO, n_splits=N_SPLITS):
    """Dobras agrupadas por ``genero_atual``; levanta erro se um gênero cair dos dois lados."""
    dobras = {s: validacao_agrupada(conj, n_splits=n_splits, seed=s) for s in seeds}
    grupos = conj["genero_atual"].to_numpy()
    for seed, divisoes in dobras.items():
        for idx_tr, idx_val in divisoes:
            comuns = set(grupos[idx_tr]) & set(grupos[idx_val])
            if comuns:
                raise ValueError(f"Gênero em treino e validação (seed {seed}): {sorted(comuns)[:5]}")
    return dobras


def cv_conjunto_final(conj, X, dobras, limiar=LIMIAR_PADRAO):
    """Pontuações fora da amostra por repetição e tabela de métricas (+ média e dp)."""
    y = conj["rotulo"].to_numpy()
    Xc = X[conj["indice"].to_numpy()]
    oof, linhas = {}, []
    for seed, divisoes in dobras.items():
        s = np.full(len(y), np.nan)
        for idx_tr, idx_val in divisoes:
            pipe = clone(construir_modelo()).fit(Xc[idx_tr], y[idx_tr])
            s[idx_val] = pontuar(pipe, Xc[idx_val])
        oof[seed] = s
        linhas.append({"repeticao": str(seed), **metricas_limiar(y, s, limiar)})
    tab = pd.DataFrame(linhas)
    num = [c for c in tab.columns if c != "repeticao"]
    resumo = pd.DataFrame([{"repeticao": "media", **tab[num].mean().to_dict()},
                           {"repeticao": "dp", **tab[num].std().to_dict()}])
    return pd.concat([tab, resumo], ignore_index=True), oof


def logo_generos(conj, X, limiar=LIMIAR_PADRAO):
    """Leave-one-genus-out dos gêneros de morcego: treina sem o gênero e pontua suas espécies."""
    linhas = []
    for gen in sorted(conj.loc[conj["rotulo"] == 1, "genero_atual"].unique()):
        fora = (conj["genero_atual"] == gen).to_numpy()
        treino = conj[~fora]
        alvo = conj[fora & (conj["rotulo"] == 1).to_numpy()]
        pipe = clone(construir_modelo()).fit(X[treino["indice"]], treino["rotulo"])
        s = pontuar(pipe, X[alvo["indice"]])
        nao = alvo.loc[s < limiar, "nome"]
        linhas.append({"genero": gen, "familia": alvo["familia"].iloc[0], "n": len(alvo),
                       "reconhecidos": int((s >= limiar).sum()),
                       "pontuacao_media": float(s.mean()), "pontuacao_min": float(s.min()),
                       "conjunto": "/".join(sorted(alvo["conjunto"].unique())),
                       "nao_reconhecidos": ";".join(nao)})
    tab = pd.DataFrame(linhas)
    tab["fracao"] = tab["reconhecidos"] / tab["n"]
    return tab.sort_values(["familia", "genero"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Erros e pontuações
# --------------------------------------------------------------------------

def vizinhos(X, ref_idx, ref_rot, nomes, alvo_idx, k=N_VIZINHOS):
    """``k`` vizinhos mais próximos (manhattan, sem escala) de cada alvo na referência."""
    ref_idx, ref_rot = np.asarray(ref_idx), np.asarray(ref_rot)
    nn = NearestNeighbors(n_neighbors=min(k, len(ref_idx)), metric=PARAMS["metric"],
                          algorithm="brute").fit(X[ref_idx])
    _, pos = nn.kneighbors(X[np.asarray(alvo_idx)])
    return [{f"viz{j}_{c}": (nomes[ref_idx[p]] if c == "nome" else int(ref_rot[p]))
             for j, p in enumerate(linha, start=1) for c in ("nome", "classe")} for linha in pos]


def tabela_erros(teste, s_teste, base, conj, oof, ann, X, limiar=LIMIAR_PADRAO):
    nomes = ann["nome"].to_numpy()
    info = ann.set_index("indice")[["genero", "familia", "ordem"]]
    t = teste.assign(pontuacao=s_teste)
    pred = t["pontuacao"] >= limiar
    neg = t[t["rotulo"] == 0].sort_values(["pontuacao", "indice"], ascending=[False, True])
    pos = t[t["rotulo"] == 1].sort_values(["pontuacao", "indice"])
    partes = [("FP", t[pred & (t["rotulo"] == 0)]), ("FN", t[~pred & (t["rotulo"] == 1)]),
              ("quase_erro_neg", neg.head(N_QUASE)), ("quase_erro_pos", pos.head(N_QUASE))]
    linhas = []
    for tipo, g in partes:
        viz = vizinhos(X, base["indice"], base["rotulo"], nomes, g["indice"]) if len(g) else []
        for (_, r), v in zip(g.iterrows(), viz):
            linhas.append({"conjunto_origem": "teste", "tipo_erro": tipo, "indice": r["indice"],
                           "rotulo": r["rotulo"], "pontuacao": r["pontuacao"], "n_rep_erro": np.nan,
                           **v})

    y = conj["rotulo"].to_numpy()
    S = np.vstack([oof[k] for k in sorted(oof)])
    erro = (S >= limiar) != (y == 1)
    n_err = erro.sum(axis=0)
    for i in np.flatnonzero(n_err):
        r = conj.iloc[i]
        ref = conj[conj["genero_atual"] != r["genero_atual"]]
        v = vizinhos(X, ref["indice"], ref["rotulo"], nomes, [r["indice"]])[0]
        linhas.append({"conjunto_origem": "validacao_agrupada",
                       "tipo_erro": "FN" if r["rotulo"] == 1 else "FP", "indice": r["indice"],
                       "rotulo": r["rotulo"], "pontuacao": float(S[:, i].mean()),
                       "n_rep_erro": int(n_err[i]), **v})
    tab = pd.DataFrame(linhas)
    if tab.empty:
        return tab
    tab = tab.join(info, on="indice")
    tab.insert(3, "nome", nomes[tab["indice"].to_numpy()])
    tab["n_rep_erro"] = tab["n_rep_erro"].astype("Int64")
    cols = ["conjunto_origem", "tipo_erro", "indice", "nome", "genero", "familia", "ordem",
            "rotulo", "pontuacao", "n_rep_erro"] + [c for c in tab.columns if c.startswith("viz")]
    return tab[cols]


def pontuacoes_todas(ann, X, pipe, base, bloqueados_idx, teste):
    """Pontuação do modelo final nas 8426 instâncias (na base, é pontuação de treino)."""
    tab = ann[["indice", "nome"]].copy()
    tab["pontuacao"] = pontuar(pipe, X[tab["indice"].to_numpy()])
    tab["na_base"] = tab["indice"].isin(base["indice"])
    tab["bloqueada"] = tab["indice"].isin(bloqueados_idx)
    tab["no_teste"] = tab["indice"].isin(teste["indice"])
    rot = pd.concat([base.set_index("indice")["rotulo"], teste.set_index("indice")["rotulo"]])
    tab["rotulo_verificado"] = tab["indice"].map(rot).astype("Int64")
    return tab


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------

def _salvar(fig, caminho):
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def figura_matriz(c, caminho):
    m = np.array([[c["vp"], c["fn"]], [c["fp"], c["vn"]]])
    fig, eixo = plt.subplots(figsize=(4.8, 4.2))
    eixo.imshow(m, cmap="Blues", vmin=0, vmax=max(m.max(), 1) * 1.6)
    for i in range(2):
        for j in range(2):
            eixo.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=14)
    eixo.set_xticks([0, 1], ["morcego", "outro"])
    eixo.set_yticks([0, 1], ["morcego", "outro"])
    eixo.set_xlabel("previsto")
    eixo.set_ylabel("real")
    eixo.set_title(f"Teste congelado (n={m.sum()}, limiar {LIMIAR_PADRAO})")
    _salvar(fig, caminho)


def figura_pontuacoes(teste, s, caminho, limiar=LIMIAR_PADRAO):
    rng = np.random.default_rng(RANDOM_STATE)
    fig, eixo = plt.subplots(figsize=(8, 3.6))
    for y0, (rot, nome, cor) in enumerate([(0, "negativos", "tab:gray"), (1, "morcegos", "tab:blue")]):
        v = s[teste["rotulo"].to_numpy() == rot]
        eixo.scatter(v, y0 + rng.uniform(-0.25, 0.25, len(v)), s=18, color=cor, alpha=0.7,
                     linewidths=0, label=f"{nome} ({len(v)})")
    eixo.axvline(limiar, color="black", linestyle="--", linewidth=1)
    eixo.text(limiar, 1.45, f" limiar {limiar}", fontsize=8, va="center")
    eixo.set_yticks([0, 1], ["negativos", "morcegos"])
    eixo.set_ylim(-0.5, 1.6)
    eixo.set_xlim(-0.05, 1.05)
    eixo.set_xlabel("pontuação do modelo final (fração ponderada dos 3 vizinhos)")
    eixo.set_title("Pontuações no teste congelado (deslocamento vertical só para leitura)")
    eixo.legend(fontsize=8, loc="center right")
    _salvar(fig, caminho)


def figura_intervalos(tab, caminho):
    tab = tab.iloc[::-1].reset_index(drop=True)
    fig, eixo = plt.subplots(figsize=(7.5, 4))
    for i, r in tab.iterrows():
        boot = r["metodo"].startswith("bootstrap")
        cor = "tab:orange" if boot else "tab:blue"
        eixo.errorbar(r["valor"], i, xerr=[[r["valor"] - r["ic_inf"]], [r["ic_sup"] - r["valor"]]],
                      fmt="s" if boot else "o", color=cor, capsize=3, markersize=6)
        eixo.text(1.02, i, f"{r['valor']:.3f} [{r['ic_inf']:.3f}; {r['ic_sup']:.3f}]",
                  va="center", fontsize=8)
    eixo.set_yticks(range(len(tab)), tab["metrica"])
    eixo.set_xlim(0, 1.45)
    eixo.set_xticks(np.linspace(0, 1, 6))
    eixo.set_xlabel("valor (IC 95%: círculo = Wilson; quadrado = bootstrap)")
    eixo.set_title("Métricas do teste congelado com intervalos de 95%")
    eixo.grid(axis="x", color="0.9")
    _salvar(fig, caminho)


def figura_logo(logo, caminho):
    tab = logo.iloc[::-1].reset_index(drop=True)
    fig, eixo = plt.subplots(figsize=(8, max(4, 0.22 * len(tab) + 1)))
    cores = np.where(tab["fracao"] >= 1, "tab:blue", np.where(tab["fracao"] > 0, "tab:orange", "tab:red"))
    eixo.barh(range(len(tab)), tab["fracao"], color=cores, height=0.7)
    for i, r in tab.iterrows():
        if r["fracao"] == 0:
            eixo.plot(0.01, i, marker="x", color="tab:red", markersize=7, clip_on=False)
        eixo.text(r["fracao"] + 0.025, i, f"{r['reconhecidos']}/{r['n']}", va="center", fontsize=7)
    eixo.set_yticks(range(len(tab)), [f"{g} ({f})" for g, f in zip(tab["genero"], tab["familia"])],
                    fontsize=7)
    eixo.set_xlim(0, 1.15)
    eixo.set_xlabel(f"fração das espécies reconhecidas (pontuação >= {LIMIAR_PADRAO})")
    eixo.set_title("Leave-one-genus-out: gêneros de morcego nunca vistos no treino\n"
                   "(azul = todas, laranja = parte, vermelho = nenhuma)")
    _salvar(fig, caminho)


# --------------------------------------------------------------------------
# Relatório
# --------------------------------------------------------------------------

def _f(v, casas=3):
    return "—" if pd.isna(v) else f"{v:.{casas}f}"


def _ic(r):
    return f"{_f(r['valor'])} [{_f(r['ic_inf'])}; {_f(r['ic_sup'])}]"


def gerar_relatorio(r):
    met, c, cv = r["metricas_teste"], r["contagens_teste"], r["cv"]
    base, conj, logo, erros = r["base"], r["conjunto"], r["logo"], r["erros"]
    por_met = met.set_index("metrica")
    n_pos, n_neg = c["vp"] + c["fn"], c["fp"] + c["vn"]
    linhas = [
        "# Relatório da Etapa 4 — modelo final e testes de performance", "",
        "Índices na base 0. O limiar e os hiperparâmetros são fixos (decididos na Etapa 2); "
        "nada aqui foi ajustado olhando o teste.", "",
        "## Configuração", "",
        f"- Classificador (`src/config_modelo.py`, origem `results/relatorio_etapa2.md`): "
        f"`{CONFIGURACAO}`. Limiar fixo: **{LIMIAR_PADRAO}**.",
        f"- Modelo final treinado com `data/rounds/rodada_{r['rodada']:02d}/base.csv` "
        f"(rodada manual {r['rodada']}, convergida): {len(base)} instâncias, "
        f"{int((base['rotulo'] == 1).sum())} morcegos e {int((base['rotulo'] == 0).sum())} negativos.",
        "", _tabela_md(base.groupby(["origem", "rotulo"]).size().rename("n").reset_index()),
        f"- Teste congelado: {n_pos + n_neg} instâncias ({n_pos} morcegos, {n_neg} negativos); "
        f"SHA256 conferido (`{r['hash_teste'][:16]}…`). Cada avaliação fica registrada em "
        f"`results/{ARQ_LOG}`.",
        "- Base final e teste verificados sem interseção de índices nem de gêneros bloqueados.", "",
        "## Teste congelado", "",
        "### Matriz de confusão", "", _tabela_md(r["matriz"]),
        "### Métricas no limiar fixo (IC 95%)", "",
        "Wilson para as proporções; bootstrap estratificado (positivos e negativos reamostrados "
        f"separadamente, {r['n_bootstrap']} vezes, `random_state=42`) para F1 e AP, que não são "
        "proporções binomiais.", "",
        _tabela_md(pd.DataFrame({
            "métrica": met["metrica"], "valor [IC 95%]": [_ic(x) for _, x in met.iterrows()],
            "k/n": [f"{k}/{n}" if pd.notna(k) else f"n={n}" for k, n in zip(met["k"], met["n"])],
            "método": met["metodo"]})),
        f"**Linha de base \"tudo 0\":** acurácia {_ic(por_met.loc['acuracia_linha_base_tudo_0'])} "
        f"sem reconhecer nenhum morcego. A acurácia do modelo "
        f"({_ic(por_met.loc['acuracia'])}) mal se distingue dela: com {n_neg} negativos em "
        f"{n_pos + n_neg}, a acurácia sozinha engana. Sensibilidade e precisão são as medidas "
        "que importam.", "",
        f"**AP e empates:** a AP ({_f(por_met.loc['precisao_media_ap', 'valor'])}) é calculada "
        "sobre pontuações com empates; negativos com a mesma pontuação de um morcego entram no "
        "mesmo degrau da curva e baixam a AP mesmo que todos os morcegos estejam acima do limiar.", "",
        "### Negativos por origem", "", _tabela_md(r["neg_origem"].assign(
            especificidade=[f"{_f(p)} [{_f(a)}; {_f(b)}]" for p, a, b in
                            zip(r["neg_origem"]["especificidade"], r["neg_origem"]["ic_inf"],
                                r["neg_origem"]["ic_sup"])]).drop(columns=["ic_inf", "ic_sup"])),
        "`negativo_provavel` **não foi verificado taxonomicamente** (gênero fora da lista de "
        "mamíferos); `mamifero_por_genero` foi.", "",
        "### Falsos positivos por ordem", "", _tabela_md(r["fp_ordem"]),
        "## Conjunto supervisionado final (base final ∪ teste)", "",
        f"{len(conj)} instâncias, {int((conj['rotulo'] == 1).sum())} morcegos de "
        f"{conj.loc[conj['rotulo'] == 1, 'genero_atual'].nunique()} gêneros. Origem dos rótulos:", "",
        _tabela_md(conj.groupby(["origem_rotulo", "rotulo"]).size().rename("n").reset_index()),
        f"**Aviso:** os {int((conj['origem_rotulo'] == 'negativo_provavel').sum())} "
        "`negativo_provavel` deste conjunto não são verificados.", "",
        "### Validação agrupada por gênero (5 dobras × 10 repetições, sementes 42..51)", "",
        "Pontuações fora da amostra por repetição; métricas no limiar fixo; AP sem limiar.", "",
        _tabela_md(cv[cv["repeticao"].isin(["media", "dp"])].set_index("repeticao")
                   [["sensibilidade", "especificidade", "precisao", "acuracia", "f1", "ap"]]
                   .T.reset_index().rename(columns={"index": "métrica"}).round(3)),
        "Matriz de confusão agregada da repetição com semente 42:", "",
        _tabela_md(matriz_confusao(cv[cv["repeticao"] == "42"].iloc[0])),
        "### Leave-one-genus-out dos gêneros de morcego", "",
        "Para cada gênero de morcego, o modelo é treinado com todo o resto do conjunto e pontua "
        "as espécies do gênero retirado: mede o reconhecimento de gêneros nunca vistos.", "",
        f"- Gêneros reconhecidos por inteiro: {int((logo['fracao'] >= 1).sum())} de {len(logo)}; "
        f"espécies reconhecidas: {int(logo['reconhecidos'].sum())} de {int(logo['n'].sum())}.", "",
        "Por família:", "",
        _tabela_md(logo.groupby("familia").agg(generos=("genero", "size"), especies=("n", "sum"),
                                               reconhecidas=("reconhecidos", "sum"),
                                               generos_inteiros=("fracao", lambda f: int((f >= 1).sum())))
                   .reset_index()),
        "Gêneros com espécies não reconhecidas:", "",
        _tabela_md(logo[logo["fracao"] < 1][["genero", "familia", "n", "reconhecidos",
                                             "pontuacao_media", "nao_reconhecidos"]].round(3)),
        "## Análise de erros", "",
        "Arquivo completo: `results/etapa4_erros.csv` (com os 3 vizinhos mais próximos: na base "
        "final para o teste; no conjunto sem o próprio gênero para a validação agrupada).", "",
    ]
    cols = ["tipo_erro", "indice", "nome", "ordem", "familia", "pontuacao", "viz1_nome",
            "viz1_classe", "viz2_nome", "viz2_classe", "viz3_nome", "viz3_classe"]
    et = erros[erros["conjunto_origem"] == "teste"] if len(erros) else erros
    for tipo, titulo in [("FP", "Falsos positivos no teste"), ("FN", "Falsos negativos no teste"),
                         ("quase_erro_neg", f"Quase-erros: {N_QUASE} negativos de maior pontuação"),
                         ("quase_erro_pos", f"Quase-erros: positivos de menor pontuação (até {N_QUASE})")]:
        g = et[et["tipo_erro"] == tipo] if len(et) else et
        linhas += [f"### {titulo}", "", _tabela_md(g[cols].round(3) if len(g) else pd.DataFrame())]
    ecv = erros[erros["conjunto_origem"] == "validacao_agrupada"] if len(erros) else erros
    linhas += ["### Erros da validação agrupada (em ≥ 1 das 10 repetições)", "",
               _tabela_md(ecv[["tipo_erro", "indice", "nome", "ordem", "familia", "pontuacao",
                               "n_rep_erro", "viz1_nome", "viz1_classe"]]
                          .sort_values(["tipo_erro", "n_rep_erro", "indice"], ascending=[True, False, True])
                          .round(3) if len(ecv) else pd.DataFrame())]
    linhas += [
        f"Nos quase-erros, negativos com a mesma pontuação estão em ordem de índice; com k=3, "
        f"{int((r['pontuacoes_teste'][r['teste']['rotulo'].to_numpy() == 0] == 0).sum())} dos "
        f"{n_neg} negativos do teste têm pontuação exatamente 0.", "",
        "## Limites", "",
        f"- **O teste tem só {n_pos} morcegos.** Cada um vale {100 / max(n_pos, 1):.0f}% da "
        f"sensibilidade; mesmo {n_pos} de {n_pos} deixa o limite inferior do IC de Wilson em "
        + f"{wilson(n_pos, n_pos)[1] * 100:.1f}%. ".replace(".", ",", 1)
        + "A sensibilidade do teste é pouco informativa sozinha.",
        f"- **Os {int((r['teste']['origem'] == 'negativo_provavel').sum())} negativos prováveis do "
        "teste não foram verificados**: se algum for morcego fora da lista, um \"falso positivo\" "
        "pode ser acerto (e vice-versa).",
        "- **O k-NN com k=3 dá pontuação discreta** (poucos valores possíveis): a AP e as curvas "
        "têm poucos degraus e o bootstrap da AP é grosseiro.",
        "- **A avaliação no conjunto supervisionado final não mede o enriquecimento sem rótulos**, "
        "só o classificador com os rótulos verificados. Os negativos prováveis desse conjunto "
        "também não são verificados.",
        "- O teste congelado é avaliado só com `--avaliar-teste`; repetir a avaliação enquanto se "
        f"altera o modelo invalida o teste. Histórico em `results/{ARQ_LOG}`.", "",
        "## Figuras", "",
    ] + [f"- `results/figuras/{f}`" for f in FIGURAS_SAIDA] + [""]
    return "\n".join(linhas)


# --------------------------------------------------------------------------
# Execução
# --------------------------------------------------------------------------

def executar(dir_seeds=DIR_SEEDS, dir_rounds=DIR_ROUNDS, dir_resultados=DIR_RESULTADOS,
             rodada_final=None, avaliar_teste=False, n_bootstrap=N_BOOTSTRAP):
    """Prepara o modelo final; com ``avaliar_teste=True`` avalia e grava as saídas."""
    dir_seeds, dir_resultados = Path(dir_seeds), Path(dir_resultados)
    df, X = carregar_tudo()
    sem = carregar_sementes(dir_seeds)
    base, n, _ = carregar_base_final(dir_rounds, rodada_final)
    bloqueados = set(instancias_bloqueadas(df, sem["arq_bloqueados"]))
    ann = anotar(df, [sem["semente"], sem["gabarito"]])
    verificar_base(base, ann, bloqueados, sem["generos_bloqueados"])
    pipe = construir_modelo().fit(X[base["indice"]], base["rotulo"])
    r = {"rodada": n, "base": base, "avaliou_teste": False,
         "resumo": {"rodada_final": n, "base_n": len(base),
                    "base_pos": int((base["rotulo"] == 1).sum()),
                    "base_neg": int((base["rotulo"] == 0).sum()),
                    "bloqueados": len(bloqueados)}}
    if not avaliar_teste:
        return r

    teste, hash_teste = ler_teste_verificado(dir_seeds)
    verificar_base(base, ann, bloqueados, sem["generos_bloqueados"], teste["indice"])
    ann = anotar(df, [sem["semente"], sem["gabarito"], teste])
    dir_figuras = dir_resultados / "figuras"
    dir_figuras.mkdir(parents=True, exist_ok=True)
    r["usos_anteriores"] = registrar_uso_teste(dir_resultados / ARQ_LOG, hash_teste, n)

    y_t = teste["rotulo"].to_numpy()
    s_t = pontuar(pipe, X[teste["indice"].to_numpy()])
    met, c = tabela_metricas_teste(y_t, s_t, n_bootstrap)
    conj = conjunto_final(base, teste, ann, sem["semente"])
    dobras = gerar_dobras(conj)
    cv, oof = cv_conjunto_final(conj, X, dobras)
    logo = logo_generos(conj, X)
    erros = tabela_erros(teste, s_t, base, conj, oof, ann, X)
    todas = pontuacoes_todas(ann, X, pipe, base, bloqueados, teste)
    r.update(avaliou_teste=True, teste=teste, hash_teste=hash_teste, pontuacoes_teste=s_t,
             metricas_teste=met, contagens_teste=c, matriz=matriz_confusao(c),
             neg_origem=negativos_por_origem(teste, s_t), fp_ordem=fp_por_ordem(teste, s_t),
             conjunto=conj, dobras=dobras, cv=cv, oof=oof, logo=logo, erros=erros, todas=todas,
             n_bootstrap=n_bootstrap)

    _csv(met, dir_resultados / "etapa4_metricas_teste.csv")
    _csv(r["matriz"], dir_resultados / "etapa4_matriz_confusao_teste.csv")
    _csv(cv, dir_resultados / "etapa4_cv_conjunto_final.csv")
    _csv(logo, dir_resultados / "etapa4_logo_generos.csv")
    _csv(erros, dir_resultados / "etapa4_erros.csv")
    _csv(todas, dir_resultados / "etapa4_pontuacoes_todas.csv")
    figura_matriz(c, dir_figuras / FIGURAS_SAIDA[0])
    figura_pontuacoes(teste, s_t, dir_figuras / FIGURAS_SAIDA[1])
    figura_intervalos(met, dir_figuras / FIGURAS_SAIDA[2])
    figura_logo(logo, dir_figuras / FIGURAS_SAIDA[3])
    (dir_resultados / "relatorio_etapa4.md").write_text(gerar_relatorio(r), encoding="utf-8",
                                                         newline="\n")
    return r


def _imprimir_resumo(r):
    res = r["resumo"]
    print(f"Rodada final: {res['rodada_final']} (convergida). Base: {res['base_n']} instâncias "
          f"({res['base_pos']} morcegos, {res['base_neg']} negativos). Sem vazamento "
          f"({res['bloqueados']} instâncias bloqueadas fora da base).")
    if not r["avaliou_teste"]:
        print("Modelo final treinado. Teste NÃO avaliado (use --avaliar-teste); nada gravado em results/.")
        return
    c = r["contagens_teste"]
    print(f"Teste: VP={c['vp']} FN={c['fn']} FP={c['fp']} VN={c['vn']}")
    for _, x in r["metricas_teste"].iterrows():
        print(f"  {x['metrica']:<28} {_ic(x)}  ({x['metodo']})")
    cv = r["cv"].set_index("repeticao")
    print("Validação agrupada no conjunto final (média ± dp):")
    for m in ("sensibilidade", "especificidade", "precisao", "acuracia", "f1", "ap"):
        print(f"  {m:<15} {cv.loc['media', m]:.3f} ± {cv.loc['dp', m]:.3f}")
    nao = r["logo"][r["logo"]["fracao"] < 1]
    print(f"LOGO: {len(nao)} gênero(s) de morcego com espécies não reconhecidas: "
          + (", ".join(nao["genero"]) or "nenhum"))
    print("Saídas gravadas em results/ (etapa4_*.csv, relatorio_etapa4.md, figuras/etapa4_*.png).")


def main():
    parser = argparse.ArgumentParser(description="Modelo final e testes de performance (Etapa 4).")
    parser.add_argument("--rodada-final", type=int, default=None,
                        help="rodada manual cuja base treina o modelo (padrão: a mais recente; precisa ter convergido)")
    parser.add_argument("--avaliar-teste", action="store_true",
                        help="avalia no teste congelado (registra o uso em results/etapa4_log_uso_teste.csv)")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()
    inicio = time.perf_counter()
    try:
        r = executar(rodada_final=args.rodada_final, avaliar_teste=args.avaliar_teste,
                     n_bootstrap=args.n_bootstrap)
    except (ValueError, RuntimeError, FileNotFoundError) as erro:
        sys.exit(f"ERRO: {erro}")
    _imprimir_resumo(r)
    print(f"Tempo: {time.perf_counter() - inicio:.1f} s")


if __name__ == "__main__":
    main()
