"""Relatório e figuras da Etapa 5 (entrega das 8426 classes).

Lê as saídas já geradas por ``scripts/gerar_classes_finais.py``
(``results/classes_finais.csv`` e ``results/indices_positivos.txt``), confere a
consistência delas e documenta de onde veio cada classe. Nada aqui altera um
rótulo nem avalia o teste congelado: o teste só é lido (pela porta com SHA256)
para saber quais índices são do teste, e nada é gravado no log de uso do teste.

A "segunda opinião" (melhor SVM RBF e melhor Random Forest da Etapa 2,
treinados na base final) é só um sinal complementar sobre o pool sem rótulo.

Uso, a partir da raiz do repositório:

    python -m src.relatorio_etapa5 --rodada-final 3
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # roda sem janela gráfica

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.avaliacao_final import (CONFIGURACAO, anotar, carregar_base_final, carregar_sementes,
                                 ler_teste_verificado, vizinhos)
from src.carregar_dados import DIR_RAIZ, N_ESPERADO, carregar_tudo, listar_nomes_fora_do_padrao
from src.config_modelo import LIMIAR_PADRAO, PARAMS
from src.enriquecimento import DIR_ROUNDS, dir_rodada
from src.experimento_etapa2 import limiar_sugerido, melhores
from src.exploracao import projetar_pca
from src.modelos import RazaoDistanciaKNN, construir_pipeline, pontuar
from src.supervisao import MOTIVO_SUSPEITO, _tabela_md, instancias_bloqueadas

RANDOM_STATE = 42
DIR_SEEDS = DIR_RAIZ / "data" / "seeds"
DIR_RESULTADOS = DIR_RAIZ / "results"

ARQ_CLASSES = "classes_finais.csv"
ARQ_LISTA = "indices_positivos.txt"
ARQ_COMPARACAO = "etapa2_comparacao.csv"
ARQ_MORCEGOS = "etapa5_morcegos.csv"
ARQ_RELATORIO = "relatorio_etapa5.md"
ARQ_UMAP_CACHE = "umap_cache.npy"
ARQ_UMAP_HASH = "umap_cache.sha256"

FONTES = ("gabarito", "teste_verificado", "base_final", "modelo_final")
ORIGEM_SEMENTE = "semente"
ORIGEM_TESTE = "teste_bloqueado"
N_TOP = 20
COR_MORCEGO, COR_MAMIFERO, COR_OUTRO = "#eb6834", "#2a78d6", "#d6d5d0"
CORES_ORIGEM = ["#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#eb6834"]  # semente, rodadas..., teste
FIGURAS = ("etapa5_fontes_rotulo.png", "etapa5_morcegos_por_familia.png",
           "etapa5_pontuacoes_sem_rotulo.png", "etapa5_mapa_pca.png")
FIGURA_UMAP = "etapa5_mapa_umap.png"


class ErroConsistencia(ValueError):
    """Os arquivos finais não estão consistentes entre si ou com as fontes."""


def _csv(tabela, caminho):
    tabela.to_csv(caminho, index=False, lineterminator="\n")


def origens_validas(rodada_final):
    return [ORIGEM_SEMENTE] + [f"rodada_{n}" for n in range(1, rodada_final + 1)] + [ORIGEM_TESTE]


# --------------------------------------------------------------------------
# Entradas e verificações de consistência
# --------------------------------------------------------------------------

def ler_lista(caminho):
    """Índices de ``indices_positivos.txt`` (um por linha), na ordem do arquivo."""
    texto = Path(caminho).read_text(encoding="utf-8")
    try:
        return [int(l) for l in texto.split()]
    except ValueError as erro:
        raise ErroConsistencia(f"{Path(caminho).name}: linha não numérica ({erro})") from None


def ler_candidatos(dir_rounds, rodada_final):
    """Todos os candidatos das rodadas manuais 1..N (``candidatos.csv``)."""
    partes = [pd.read_csv(dir_rodada(dir_rounds, n) / "candidatos.csv").assign(rodada=n)
              for n in range(1, rodada_final + 1)]
    return pd.concat(partes, ignore_index=True)


def verificar_consistencia(classes, lista, base_lista, proibidos, candidatos, base):
    """Levanta ``ErroConsistencia`` na primeira regra violada; devolve as contagens por fonte."""
    if len(classes) != N_ESPERADO:
        raise ErroConsistencia(f"{ARQ_CLASSES}: {len(classes)} linhas (esperado {N_ESPERADO}).")
    if not classes["indice_base0"].is_unique:
        raise ErroConsistencia(f"{ARQ_CLASSES}: índices repetidos.")
    if sorted(classes["indice_base0"]) != list(range(N_ESPERADO)):
        raise ErroConsistencia(f"{ARQ_CLASSES}: indice_base0 não cobre 0..{N_ESPERADO - 1}.")
    if not (classes["indice_base1"] == classes["indice_base0"] + 1).all():
        raise ErroConsistencia(f"{ARQ_CLASSES}: indice_base1 != indice_base0 + 1.")
    valores = set(classes["classe"].unique())
    if not valores <= {0, 1}:
        raise ErroConsistencia(f"{ARQ_CLASSES}: classes fora de {{0, 1}}: {sorted(valores - {0, 1})}.")

    positivos = classes.loc[classes["classe"] == 1]
    esperada = sorted(positivos[f"indice_base{base_lista}"])
    if sorted(lista) != esperada or len(lista) != len(set(lista)):
        outra = 1 - base_lista
        dica = (f" A lista confere com a base {outra}: a base declarada está errada."
                if sorted(lista) == sorted(positivos[f"indice_base{outra}"]) else "")
        raise ErroConsistencia(f"{ARQ_LISTA} ({len(lista)} índices) não tem exatamente os "
                               f"{len(esperada)} positivos de {ARQ_CLASSES} na base {base_lista}.{dica}")

    fontes = set(classes["fonte_rotulo"])
    if not fontes <= set(FONTES):
        raise ErroConsistencia(f"{ARQ_CLASSES}: fonte_rotulo desconhecida: {sorted(fontes - set(FONTES))}.")
    fora_gab = positivos[positivos["fonte_rotulo"] != "gabarito"]
    if len(fora_gab):
        raise ErroConsistencia(f"{len(fora_gab)} positivo(s) não vêm do gabarito: "
                               f"{fora_gab['indice_base0'].tolist()[:10]} "
                               f"(fontes {sorted(set(fora_gab['fonte_rotulo']))}).")
    contagem = classes["fonte_rotulo"].value_counts()
    if int(contagem.sum()) != N_ESPERADO:
        raise ErroConsistencia(f"Soma por fonte = {int(contagem.sum())} (esperado {N_ESPERADO}).")

    for onde, indices in (("candidatos das rodadas manuais", candidatos["indice"]),
                          ("base final", base["indice"])):
        vazados = sorted(set(map(int, indices)) & set(proibidos))
        if vazados:
            raise ErroConsistencia(f"{len(vazados)} índice(s) do teste/gêneros bloqueados em {onde}: "
                                   f"{vazados[:10]}.")
    return pd.crosstab(classes["fonte_rotulo"], classes["classe"]).reindex(
        index=list(FONTES), columns=[0, 1], fill_value=0)


# --------------------------------------------------------------------------
# Morcegos
# --------------------------------------------------------------------------

def montar_morcegos(classes, gabarito, base, proibidos, rodada_final):
    """Uma linha por morcego, com gênero/família do gabarito e a origem da entrada."""
    pos = classes.loc[classes["classe"] == 1, ["indice_base0", "indice_base1", "nome"]]
    gab = gabarito.set_index(gabarito["indice"].astype(int))
    faltam = sorted(set(pos["indice_base0"]) - set(gab.index))
    if faltam:
        raise ErroConsistencia(f"Morcegos fora do gabarito: {faltam[:10]}.")
    pos = pos.assign(genero=pos["indice_base0"].map(gab["genero_atual"]),
                     familia=pos["indice_base0"].map(gab["familia"]))
    rodada = base.set_index("indice")["rodada"]
    origem = []
    for i in pos["indice_base0"]:
        if i in rodada.index:
            r = int(rodada[i])
            origem.append(ORIGEM_SEMENTE if r == 0 else f"rodada_{r}")
        elif i in proibidos:
            origem.append(ORIGEM_TESTE)
        else:
            raise ErroConsistencia(f"Morcego {i} não está na base final nem no teste/gêneros bloqueados.")
    pos["origem_entrada"] = origem
    invalidas = set(origem) - set(origens_validas(rodada_final))
    if invalidas:
        raise ErroConsistencia(f"origem_entrada inválida: {sorted(invalidas)}.")
    return pos.sort_values("indice_base0").reset_index(drop=True)


# --------------------------------------------------------------------------
# Instâncias sem rótulo
# --------------------------------------------------------------------------

def proximidade_morcego(X, base, alvo):
    """``d_neg / (d_pos + d_neg)`` (manhattan) na base final: só para desempatar a ordenação."""
    modelo = RazaoDistanciaKNN(metric=PARAMS["metric"]).fit(X[base["indice"]], base["rotulo"].to_numpy())
    return modelo.decision_function(X[np.asarray(alvo)])


def analisar_sem_rotulo(classes, ann, X, base, bloqueados, suspeitos, limiar):
    """Tabela das instâncias decididas pelo modelo, com gênero, bloqueio e proximidade."""
    sr = classes.loc[classes["fonte_rotulo"] == "modelo_final",
                     ["indice_base0", "indice_base1", "nome", "classe", "pontuacao"]].copy()
    info = ann.set_index("indice")
    sr["genero"] = sr["indice_base0"].map(info["genero_atual"])
    sr["genero_bloqueado"] = sr["indice_base0"].isin(bloqueados)
    sr["genero_suspeito"] = sr["indice_base0"].isin(suspeitos)
    sr["proximidade"] = proximidade_morcego(X, base, sr["indice_base0"])
    sr = sr.sort_values(["pontuacao", "proximidade", "indice_base0"],
                        ascending=[False, False, True]).reset_index(drop=True)
    distrib = {"exatamente_0": int((sr["pontuacao"] == 0).sum()),
               "entre_0_e_limiar": int(((sr["pontuacao"] > 0) & (sr["pontuacao"] < limiar)).sum()),
               "acima_limiar": int((sr["pontuacao"] >= limiar).sum())}
    top = sr.head(N_TOP).copy()
    viz = vizinhos(X, base["indice"], base["rotulo"], ann["nome"].to_numpy(), top["indice_base0"])
    top = pd.concat([top.reset_index(drop=True), pd.DataFrame(viz)], axis=1)
    return sr, distrib, top


def segunda_opiniao(comparacao, X, base, sr, top_knn):
    """Melhor SVM RBF e melhor RF da Etapa 2, treinados na base final, pontuando o pool sem rótulo."""
    melhor = melhores(comparacao)
    alvo = sr["indice_base0"].to_numpy()
    top_ids = set(top_knn["indice_base0"])
    saida = {}
    for modelo, nome in (("svm", "SVM RBF"), ("rf", "Random Forest")):
        linha = melhor.loc[modelo]
        pipe = construir_pipeline(linha["variante"], json.loads(linha["params"]), bool(linha["escala"]))
        pipe.fit(X[base["indice"]], base["rotulo"].to_numpy())
        s = pontuar(pipe, X[alvo])
        limiar, origem = limiar_sugerido(linha)
        tab = sr[["indice_base0", "nome", "genero", "genero_bloqueado"]].assign(pontuacao=s)
        tab = tab.sort_values(["pontuacao", "indice_base0"], ascending=[False, True]).head(N_TOP)
        tab["no_top_knn"] = tab["indice_base0"].isin(top_ids)
        saida[modelo] = {"nome": nome, "config": linha["config_id"], "limiar": limiar,
                         "origem_limiar": origem, "n_acima": int((s >= limiar).sum()),
                         "max": float(s.max()), "top": tab.reset_index(drop=True),
                         "intersecao": sorted(set(tab["indice_base0"]) & top_ids)}
    return saida


# --------------------------------------------------------------------------
# UMAP em cache (nunca calculado aqui)
# --------------------------------------------------------------------------

def carregar_umap_cache(dir_resultados):
    """``(Y, motivo)``: Y só se ``umap_cache.npy`` existir, o SHA256 conferir e a forma for (8426, 2)."""
    arq, arq_hash = Path(dir_resultados) / ARQ_UMAP_CACHE, Path(dir_resultados) / ARQ_UMAP_HASH
    if not arq.exists():
        return None, f"`results/{ARQ_UMAP_CACHE}` não existe"
    if not arq_hash.exists():
        return None, f"`results/{ARQ_UMAP_HASH}` não existe (hash não conferível)"
    esperado = arq_hash.read_text(encoding="utf-8").split()[0]
    if hashlib.sha256(arq.read_bytes()).hexdigest() != esperado:
        return None, f"SHA256 de `results/{ARQ_UMAP_CACHE}` não confere"
    Y = np.load(arq)
    if Y.shape != (N_ESPERADO, 2):
        return None, f"`results/{ARQ_UMAP_CACHE}` com forma {Y.shape} (esperado ({N_ESPERADO}, 2))"
    return Y, "cache conferido"


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------

def _salvar(fig, caminho):
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def _eixo_limpo(eixo):
    for lado in ("top", "right"):
        eixo.spines[lado].set_visible(False)


def figura_fontes(tab, caminho):
    rotulos = {"gabarito": "gabarito\n(taxonomia)", "teste_verificado": "teste verificado",
               "base_final": "base final", "modelo_final": "modelo final\n(sem rótulo)"}
    fontes = list(tab.index)[::-1]
    y = np.arange(len(fontes))
    fig, eixo = plt.subplots(figsize=(8, 4))
    n0, n1 = tab.loc[fontes, 0].to_numpy(), tab.loc[fontes, 1].to_numpy()
    eixo.barh(y, n0, color=COR_MAMIFERO, height=0.6, label="0 = outro", edgecolor="white", linewidth=2)
    eixo.barh(y, n1, left=n0, color=COR_MORCEGO, height=0.6, label="1 = morcego",
              edgecolor="white", linewidth=2)
    for yi, a, b in zip(y, n0, n1):
        texto = f"{a} (0)" + (f" + {b} (1)" if b else "")
        eixo.text(a + b + 60, yi, texto, va="center", fontsize=9)
    eixo.set_yticks(y, [rotulos[f] for f in fontes])
    eixo.set_xlim(0, (n0 + n1).max() * 1.25)
    eixo.set_xlabel("instâncias")
    eixo.set_title(f"Fonte do rótulo final x classe ({int(tab.to_numpy().sum())} instâncias)")
    eixo.legend(loc="upper right", frameon=False)
    _eixo_limpo(eixo)
    _salvar(fig, caminho)


def figura_familias(morcegos, origens, caminho):
    rot = {o: ("teste (gênero bloqueado)" if o == ORIGEM_TESTE else o.replace("_", " ")) for o in origens}
    tab = pd.crosstab(morcegos["familia"], morcegos["origem_entrada"]).reindex(columns=origens, fill_value=0)
    tab = tab.loc[:, tab.sum() > 0]
    tab = tab.loc[tab.sum(axis=1).sort_values().index]
    y = np.arange(len(tab))
    fig, eixo = plt.subplots(figsize=(8, 5))
    esquerda = np.zeros(len(tab))
    for j, o in enumerate(tab.columns):
        eixo.barh(y, tab[o], left=esquerda, color=CORES_ORIGEM[origens.index(o) % len(CORES_ORIGEM)],
                  height=0.65, label=rot[o], edgecolor="white", linewidth=2)
        esquerda += tab[o].to_numpy()
    for yi, total in zip(y, esquerda):
        eixo.text(total + 0.5, yi, str(int(total)), va="center", fontsize=9)
    eixo.set_yticks(y, tab.index)
    eixo.xaxis.get_major_locator().set_params(integer=True)
    eixo.set_xlim(0, esquerda.max() * 1.12)
    eixo.set_xlabel("morcegos")
    eixo.set_title(f"Morcegos por família e origem da entrada ({len(morcegos)} no total)")
    eixo.legend(loc="lower right", frameon=False, title="origem")
    _eixo_limpo(eixo)
    _salvar(fig, caminho)


def figura_pontuacoes(sr, distrib, limiar, caminho):
    fig, (e1, e2) = plt.subplots(1, 2, figsize=(11, 4.2))
    e1.hist(sr["pontuacao"], bins=np.linspace(0, 1, 21), color=COR_MAMIFERO, edgecolor="white")
    e1.set_yscale("log")
    e1.axvline(limiar, color="#0b0b0b", linestyle="--", linewidth=1.2, label=f"limiar {limiar}")
    n0 = max(distrib["exatamente_0"], 1)
    e1.set_ylim(0.8, n0 * 20)
    e1.annotate(f"{distrib['exatamente_0']} com pontuação exatamente 0", xy=(0.05, n0),
                xytext=(0.12, n0 * 6), fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "#52514e"})
    e1.set_xlim(0, 1)
    e1.set_xlabel("pontuação do k-NN final")
    e1.set_ylabel("instâncias (escala log)")
    e1.set_title(f"Pontuações das {len(sr)} instâncias sem rótulo")
    e1.legend(frameon=False, loc="center right")
    e2.hist(sr["proximidade"], bins=40, color=COR_MAMIFERO, edgecolor="white")
    e2.set_yscale("log")
    e2.axvline(0.5, color="#52514e", linestyle=":", linewidth=1.2, label="igual distância (0,5)")
    e2.set_xlabel("d_neg / (d_pos + d_neg) na base final")
    e2.set_ylabel("instâncias (escala log)")
    e2.set_title("Auxiliar (só desempate): proximidade de morcegos")
    e2.legend(frameon=False, loc="upper left")
    for e in (e1, e2):
        _eixo_limpo(e)
    _salvar(fig, caminho)


def figura_mapa(Y, classes, gabarito, titulo, eixos, caminho):
    pos = classes.loc[classes["classe"] == 1, "indice_base0"].to_numpy()
    mam = gabarito.loc[gabarito["rotulo_gabarito"].astype(int) == 0, "indice"].astype(int).to_numpy()
    resto = np.setdiff1d(np.arange(len(Y)), np.concatenate([pos, mam]))
    fig, eixo = plt.subplots(figsize=(8, 7))
    eixo.scatter(Y[resto, 0], Y[resto, 1], s=3, color=COR_OUTRO, linewidths=0,
                 label=f"demais organismos ({len(resto)})")
    eixo.scatter(Y[mam, 0], Y[mam, 1], s=9, color=COR_MAMIFERO, linewidths=0,
                 label=f"mamíferos não-morcegos verificados ({len(mam)})")
    eixo.scatter(Y[pos, 0], Y[pos, 1], s=26, color=COR_MORCEGO, edgecolors="white", linewidths=0.6,
                 label=f"morcegos ({len(pos)})")
    eixo.set_xlabel(eixos[0])
    eixo.set_ylabel(eixos[1])
    eixo.set_title(titulo)
    eixo.legend(frameon=False, markerscale=1.5, loc="lower left")
    _eixo_limpo(eixo)
    _salvar(fig, caminho)


# --------------------------------------------------------------------------
# Relatório
# --------------------------------------------------------------------------

def _origens_texto(g, origens):
    cont = g["origem_entrada"].value_counts()
    return "; ".join(f"{o.replace('_', ' ')} {cont[o]}" for o in origens if o in cont)


def _vizinhos_texto(r):
    return "; ".join(f"{r[f'viz{j}_nome']} ({r[f'viz{j}_classe']})" for j in range(1, 4))


def _cod(texto):
    """Código numa célula de tabela Markdown (``|`` escapado)."""
    return "`" + str(texto).replace("|", r"\|") + "`"


def _sim(v):
    return "sim" if v else "não"


def gerar_relatorio(r):
    tab, mor, origens = r["tab_fontes"], r["morcegos"], r["origens"]
    tab_md = tab.reset_index().rename(columns={"fonte_rotulo": "fonte", 0: "classe 0", 1: "classe 1"})
    tab_md["total"] = tab_md["classe 0"] + tab_md["classe 1"]
    tab_md.loc[len(tab_md)] = ["total", int(tab[0].sum()), int(tab[1].sum()), int(tab.to_numpy().sum())]
    n_pos, b = len(mor), r["base_lista"]

    fam = mor.groupby("familia").apply(lambda g: pd.Series({
        "generos": g["genero"].nunique(), "morcegos": len(g), "origem": _origens_texto(g, origens)}),
        include_groups=False).reset_index().sort_values(["morcegos", "familia"], ascending=[False, True])
    gen = mor.groupby(["familia", "genero"]).apply(lambda g: pd.Series({
        "n": len(g), "origem": _origens_texto(g, origens),
        "especies": ", ".join(g["nome"])}), include_groups=False).reset_index()
    orig = mor["origem_entrada"].value_counts().reindex(origens, fill_value=0).reset_index()
    orig.columns = ["origem_entrada", "morcegos"]

    d, sr = r["distrib"], r["sem_rotulo"]
    top = r["top_knn"].copy()
    top["vizinhos na base final (classe)"] = top.apply(_vizinhos_texto, axis=1)
    top["gênero bloqueado"] = top["genero_bloqueado"].map(_sim)
    top["proximidade"] = top["proximidade"].round(4)
    top_md = top[["indice_base0", "nome", "genero", "pontuacao", "proximidade", "gênero bloqueado",
                  "vizinhos na base final (classe)"]]
    bloq = sr[sr["genero_bloqueado"]].sort_values("indice_base0")
    susp = sr[sr["genero_suspeito"]].sort_values("indice_base0")
    disc = r["discordancias"]

    L = [
        "# Relatório da Etapa 5 — entrega das 8426 classes", "",
        "Este relatório não altera nenhum rótulo e não avalia o teste congelado: nenhuma métrica é "
        "calculada sobre o teste e nada é gravado em `results/etapa4_log_uso_teste.csv`. O teste só é "
        "lido (com o SHA256 conferido) para saber quais índices pertencem a ele.", "",
        "## Configuração", "",
        f"- Arquivos entregues: `results/{ARQ_CLASSES}` (8426 linhas: `indice_base0`, `indice_base1`, "
        f"`nome`, `classe`, `fonte_rotulo`, `pontuacao`) e `results/{ARQ_LISTA}` ({n_pos} linhas), "
        "gerados por `scripts/gerar_classes_finais.py`.",
        f"- Modelo final: `{CONFIGURACAO}` (`src/config_modelo.py`), treinado na base da rodada manual "
        f"{r['rodada']} (convergida, `data/rounds/rodada_{r['rodada']:02d}/base.csv`: {r['n_base']} "
        f"instâncias, {r['n_base_pos']} morcegos). Limiar fixo: **{r['limiar']}** (Etapa 2).",
        "- Prioridade do rótulo: gabarito taxonômico → teste verificado → base final → modelo final "
        "(pontuação ≥ limiar). O modelo só decide onde não há rótulo verificado.",
        f"- **Base dos índices:** o enunciado pede os *números de ordem* dos morcegos, contados a partir "
        f"de 1 (a primeira linha de `Names.txt` é o organismo 1). Por isso `{ARQ_LISTA}` usa a **base "
        f"{b}** (`indice_base{b}`). Internamente o projeto usa base 0 (`indice` = `indice_base0`); "
        "`indice_base1 = indice_base0 + 1`. As tabelas abaixo mostram `indice_base0`.", "",
        "## Verificações de consistência", "",
        "Todas passaram (o comando para com erro se alguma falhar):", "",
        f"- {N_ESPERADO} linhas, `indice_base0` único e cobrindo 0..{N_ESPERADO - 1}; "
        "`indice_base1 = indice_base0 + 1`;",
        "- `classe` só 0 ou 1;",
        f"- `{ARQ_LISTA}` tem exatamente os {n_pos} positivos de `{ARQ_CLASSES}` na base {b};",
        f"- os {n_pos} positivos vêm do gabarito; a soma por fonte é {N_ESPERADO};",
        f"- nenhum índice do teste ou de gênero bloqueado ({r['n_proibidos']} índices) aparece nos "
        f"candidatos das rodadas manuais 1..{r['rodada']} ({r['n_candidatos']} candidatos) nem na base final;",
        "- as contagens da tabela abaixo são relidas deste relatório e comparadas com o CSV.", "",
        "## Fonte do rótulo x classe", "",
        _tabela_md(tab_md),
        f"**{n_pos} morcegos**, todos com rótulo do gabarito taxonômico (MDD v2.5, ordem Chiroptera). "
        f"Positivos decididos só pelo modelo: **{r['n_pos_modelo']}**.", "",
        "## Morcegos por família e gênero", "",
        f"Arquivo: `results/{ARQ_MORCEGOS}` (uma linha por morcego; gênero e família do gabarito). "
        "`origem_entrada` = quando o morcego entrou na base de treino: `semente` (rodada 0), "
        f"`rodada_N` (coluna `rodada` de `data/rounds/rodada_{r['rodada']:02d}/base.csv`) ou "
        f"`{ORIGEM_TESTE}` (teste congelado; nunca entrou no treino).", "",
        _tabela_md(orig), "### Por família", "", _tabela_md(fam), "### Por gênero", "", _tabela_md(gen),
        "## O que o modelo decidiu", "",
        f"O modelo final decidiu sozinho {len(sr)} instâncias (sem rótulo em nenhuma fonte verificada). "
        f"**Nenhuma passou do limiar: zero positivos vieram só do modelo.**", "",
        "| pontuação | instâncias |", "|---|---|",
        f"| exatamente 0 | {d['exatamente_0']} |",
        f"| entre 0 e o limiar ({r['limiar']}) | {d['entre_0_e_limiar']} |",
        f"| ≥ limiar | {d['acima_limiar']} |", "",
        "Com k=3 e votos por distância, a pontuação só é maior que 0 quando ao menos um dos 3 vizinhos "
        "mais próximos na base final é morcego. Como todas as pontuações são iguais, a ordem das "
        f"\"{N_TOP} de maior pontuação\" foi desempatada pela proximidade "
        "`d_neg / (d_pos + d_neg)` (manhattan, sem escala, a variante `knn_razao` da Etapa 2): "
        "d_pos e d_neg são as distâncias ao morcego e ao não-morcego mais próximos da base final; "
        "1 = em cima de um morcego, 0,5 = a igual distância. **Esse desempate só ordena a tabela; não "
        "muda nenhuma classe.**", "",
        f"### As {N_TOP} sem rótulo de maior pontuação (desempate pela proximidade)", "",
        _tabela_md(top_md),
        f"Proximidade máxima no pool: {sr['proximidade'].max():.4f}; instâncias com proximidade > 0,5: "
        f"{int((sr['proximidade'] > 0.5).sum())}. Valores perto de 0,5 aparecem em organismos isolados, "
        "longe de toda a base (morcegos e não-morcegos quase à mesma distância), e não indicam semelhança "
        f"com morcegos. Entre as {N_TOP}, {r['top_com_viz_morcego']} têm algum morcego entre os 3 vizinhos. "
        "Gênero bloqueado = gênero reservado "
        "ao teste.", "",
        "### Sem rótulo em gêneros bloqueados", "",
        f"Os gêneros do teste somam {r['n_bloqueados']} instâncias: {r['bloq_fontes'].get('gabarito', 0)} "
        f"com rótulo do gabarito, {r['bloq_fontes'].get('teste_verificado', 0)} negativos prováveis do "
        f"teste e **{len(bloq)} sem rótulo**, decididas pelo modelo (todas fora do treino e do "
        "enriquecimento):", "",
        _tabela_md(bloq[["indice_base0", "nome", "genero", "pontuacao", "proximidade"]].round(4)),
        "### Sem rótulo com gênero suspeito", "",
        f"{len(susp)} nomes cujo primeiro termo está em minúscula (não é gênero); nunca receberam rótulo:", "",
        _tabela_md(susp[["indice_base0", "nome", "pontuacao", "proximidade"]].round(4)),
        "## Rótulo verificado contra o modelo", "",
        f"{len(disc)} instância(s) em que o rótulo verificado discorda do modelo no limiar:", "",
        _tabela_md(disc),
    ]
    if (disc["nome"] == "Galemys_pyrenaicus").any():
        L += ["O caso de *Galemys pyrenaicus* (desmã-dos-pirenéus, Eulipotyphla: Talpidae) mostra a "
              "prioridade funcionando: o modelo dá pontuação 1,0 porque os 3 vizinhos mais próximos na "
              "base final são morcegos, mas o gabarito taxonômico diz que não é morcego, e a classe "
              "final segue o gabarito (0). É o mesmo falso positivo do teste na Etapa 4.", ""]

    L += ["## Segunda opinião (complementar)", "",
          "**Isso é só um sinal adicional de ausência de candidatos escondidos; nenhum rótulo é alterado.** "
          "A melhor configuração de cada modelo na Etapa 2 (`results/etapa2_comparacao.csv`, maior AP, "
          "mesma regra de desempate da Etapa 2) foi treinada na base final e pontuou as "
          f"{len(sr)} instâncias sem rótulo. O limiar é o sugerido na Etapa 2 (mediana do menor limiar com "
          "precisão ≥ 0,95), que só transfere de forma aproximada para um modelo treinado em outra base "
          "(sobretudo o SVM, cuja `decision_function` muda de escala a cada ajuste).", "",
          "| modelo | configuração | limiar sugerido | sem rótulo ≥ limiar | pontuação máxima | "
          f"interseção com o top {N_TOP} do k-NN |", "|---|---|---|---|---|---|"]
    for o in r["segunda"].values():
        L.append(f"| {o['nome']} | {_cod(o['config'])} | {o['limiar']:.4g} | {o['n_acima']} | "
                 f"{o['max']:.4g} | {len(o['intersecao'])} |")
    L.append("")
    for o in r["segunda"].values():
        t = o["top"].copy()
        t["pontuacao"] = t["pontuacao"].round(4)
        t["gênero bloqueado"] = t["genero_bloqueado"].map(_sim)
        t["no top k-NN"] = t["no_top_knn"].map(_sim)
        L += [f"### {o['nome']}: as {N_TOP} sem rótulo de maior pontuação", "",
              _tabela_md(t[["indice_base0", "nome", "genero", "pontuacao", "gênero bloqueado", "no top k-NN"]])]

    umap = (f"- `results/figuras/{FIGURA_UMAP}`" if r["umap_ok"]
            else f"- UMAP não gerado: {r['umap_motivo']}. O UMAP não é recalculado na Etapa 5 "
                 "(a projeção já causou MemoryError na máquina da equipe).")
    L += [
        "## Limites", "",
        "- **A classificação final é taxonomia onde ela existe e modelo no restante.** Os 79 morcegos "
        "vêm todos da lista taxonômica; o modelo não acrescentou nenhum.",
        f"- **Os {len(sr)} organismos sem rótulo ficaram com classe 0 sem verificação taxonômica.** "
        "A decisão é só do modelo, e todas as pontuações são 0.",
        f"- **Os {r['n_prov']} negativos prováveis ({r['n_prov_base']} na base e {r['n_prov_teste']} no "
        "teste) não foram verificados**: foram sorteados entre gêneros fora da lista de mamíferos.",
        "- **Ausência de candidatos não prova ausência de morcegos fora do MDD.** Um morcego cujo gênero "
        "não casou com a lista ficaria sem rótulo e, se o modelo não o reconhecesse, seria 0.",
        "- Nomes de gênero antigos ou grafias diferentes podem não ter casado com a lista; os gêneros da "
        "lista sem correspondência em `Names.txt` estão em `results/generos_lista_sem_correspondencia.csv`.",
        "- A contribuição do aprendizado de máquina está na simulação do oculto (Etapa 3: recuperação de "
        f"{r['oculto_achados']} de {r['oculto_total']} morcegos do oculto nas rodadas manuais, incluindo famílias sem representante na semente), na verificação "
        "do pool sem rótulo (aqui, com a segunda opinião) e na avaliação (Etapa 4). Na lista entregue, "
        "a contribuição direta do modelo é nula: nenhum positivo novo.", "",
        "## Figuras", "",
        *[f"- `results/figuras/{f}`" for f in FIGURAS], umap, "",
        "## Como reproduzir", "",
        "Na raiz do repositório, com o ambiente ativado, nesta ordem:", "",
        "```bash",
        "python -m src.exploracao",
        "python -m src.supervisao",
        "python -m src.experimento_etapa2",
        "python -m src.enriquecimento --rodada 1   # e 2, 3, preenchendo revisao_manual.csv entre elas",
        "python -m src.avaliacao_final --rodada-final 3 --avaliar-teste",
        "python scripts/gerar_classes_finais.py --rodada-final 3 --base-lista 1",
        "python -m src.relatorio_etapa5 --rodada-final 3",
        "```", "",
        "O teste congelado só é avaliado com `--avaliar-teste` (Etapa 4), e cada uso fica registrado em "
        "`results/etapa4_log_uso_teste.csv`. A Etapa 5 não reavalia o teste. Tudo usa `random_state=42`; "
        "repetir os comandos dá as mesmas saídas.", "",
    ]
    return "\n".join(L)


_LINHA_FONTE = re.compile(r"^\| (" + "|".join(FONTES) + r") \| (\d+) \| (\d+) \| (\d+) \|$", re.M)


def verificar_contagens_relatorio(texto, classes):
    """Relê a tabela fonte x classe do relatório e compara com o CSV."""
    achadas = {f: (int(a), int(b), int(t)) for f, a, b, t in _LINHA_FONTE.findall(texto)}
    if set(achadas) != set(FONTES):
        raise ErroConsistencia(f"Tabela fonte x classe incompleta no relatório: {sorted(achadas)}.")
    tab = pd.crosstab(classes["fonte_rotulo"], classes["classe"]).reindex(
        index=list(FONTES), columns=[0, 1], fill_value=0)
    for f, (a, b, t) in achadas.items():
        if (a, b, t) != (int(tab.loc[f, 0]), int(tab.loc[f, 1]), int(tab.loc[f].sum())):
            raise ErroConsistencia(f"Relatório diverge do CSV na fonte {f}: {(a, b, t)}.")
    m = re.search(r"\*\*(\d+) morcegos\*\*", texto)
    if not m or int(m.group(1)) != int((classes["classe"] == 1).sum()):
        raise ErroConsistencia("Número de morcegos no relatório diverge do CSV.")


# --------------------------------------------------------------------------
# Execução
# --------------------------------------------------------------------------

def executar(rodada_final, base_lista=1, dir_seeds=DIR_SEEDS, dir_rounds=DIR_ROUNDS,
             dir_entrada=DIR_RESULTADOS, dir_saida=DIR_RESULTADOS, limiar=LIMIAR_PADRAO):
    """Confere os arquivos finais e grava ``etapa5_morcegos.csv``, o relatório e as figuras.

    ``dir_entrada`` tem ``classes_finais.csv``, ``indices_positivos.txt``,
    ``etapa2_comparacao.csv`` e, opcionalmente, o UMAP em cache.
    """
    dir_entrada, dir_saida = Path(dir_entrada), Path(dir_saida)
    classes = pd.read_csv(dir_entrada / ARQ_CLASSES)
    lista = ler_lista(dir_entrada / ARQ_LISTA)
    base, n, met = carregar_base_final(dir_rounds, rodada_final)
    sementes = carregar_sementes(dir_seeds)
    gabarito = pd.read_csv(Path(dir_seeds) / "gabarito.csv", keep_default_na=False)
    teste, _ = ler_teste_verificado(dir_seeds)  # só leitura; não registra uso nem calcula métrica
    df, X = carregar_tudo()
    bloqueados = set(instancias_bloqueadas(df, sementes["arq_bloqueados"]))
    proibidos = bloqueados | set(teste["indice"])
    candidatos = ler_candidatos(dir_rounds, n)

    tab = verificar_consistencia(classes, lista, base_lista, proibidos, candidatos, base)
    morcegos = montar_morcegos(classes, gabarito, base, proibidos, n)

    ann = anotar(df, [sementes["semente"], gabarito, teste])
    fora = listar_nomes_fora_do_padrao(df)
    suspeitos = set(fora.loc[fora["motivo"].str.contains(MOTIVO_SUSPEITO, regex=False), "indice"])
    sr, distrib, top = analisar_sem_rotulo(classes, ann, X, base, bloqueados, suspeitos, limiar)
    segunda = segunda_opiniao(pd.read_csv(dir_entrada / ARQ_COMPARACAO), X, base, sr, top)

    verif = classes[classes["fonte_rotulo"] != "modelo_final"]
    disc = verif[(verif["pontuacao"] >= limiar) != (verif["classe"] == 1)].copy()
    if len(disc):
        info = ann.set_index("indice")
        disc["ordem"] = disc["indice_base0"].map(info["ordem"])
        disc["familia"] = disc["indice_base0"].map(info["familia"])
        v = pd.DataFrame(vizinhos(X, base["indice"], base["rotulo"], ann["nome"].to_numpy(),
                                  disc["indice_base0"]), index=disc.index)
        disc["vizinhos na base final (classe)"] = v.apply(_vizinhos_texto, axis=1)
    disc = disc.drop(columns=["indice_base1"])

    semente = sementes["semente"]
    prov = semente["origem"] == "negativo_provavel"
    r = {"rodada": n, "base_lista": base_lista, "limiar": limiar, "tab_fontes": tab,
         "morcegos": morcegos, "origens": origens_validas(n), "sem_rotulo": sr, "distrib": distrib,
         "top_knn": top, "segunda": segunda, "discordancias": disc,
         "n_base": len(base), "n_base_pos": int(base["rotulo"].sum()),
         "n_proibidos": len(proibidos), "n_candidatos": len(candidatos),
         "n_pos_modelo": int(((classes["fonte_rotulo"] == "modelo_final") & (classes["classe"] == 1)).sum()),
         "n_bloqueados": len(bloqueados), "oculto_achados": met["morcegos_oculto_encontrados_acum"],
         "oculto_total": met["morcegos_oculto_total"],
         "top_com_viz_morcego": int((top[[f"viz{j}_classe" for j in (1, 2, 3)]] == 1).any(axis=1).sum()),
         "bloq_fontes": classes[classes["indice_base0"].isin(bloqueados)]["fonte_rotulo"].value_counts().to_dict(),
         "n_prov_base": int(base["indice"].isin(semente.loc[prov, "indice"]).sum()),
         "n_prov_teste": int((teste["origem"] == "negativo_provavel").sum())}
    r["n_prov"] = r["n_prov_base"] + r["n_prov_teste"]

    dir_fig = dir_saida / "figuras"
    dir_fig.mkdir(parents=True, exist_ok=True)
    figura_fontes(tab, dir_fig / FIGURAS[0])
    figura_familias(morcegos, r["origens"], dir_fig / FIGURAS[1])
    figura_pontuacoes(sr, distrib, limiar, dir_fig / FIGURAS[2])
    Y, var = projetar_pca(X)
    figura_mapa(Y, classes, gabarito, f"PCA 2D de todos os organismos (variância explicada: "
                f"{var[0]:.1%} + {var[1]:.1%} = {var.sum():.1%})", ("componente 1", "componente 2"),
                dir_fig / FIGURAS[3])
    Y_umap, r["umap_motivo"] = carregar_umap_cache(dir_entrada)
    r["umap_ok"] = Y_umap is not None
    if r["umap_ok"]:
        figura_mapa(Y_umap, classes, gabarito, "UMAP 2D em cache (random_state=42)",
                    ("UMAP 1", "UMAP 2"), dir_fig / FIGURA_UMAP)

    texto = gerar_relatorio(r)
    verificar_contagens_relatorio(texto, classes)
    _csv(morcegos[["indice_base0", "indice_base1", "nome", "genero", "familia", "origem_entrada"]],
         dir_saida / ARQ_MORCEGOS)
    (dir_saida / ARQ_RELATORIO).write_text(texto, encoding="utf-8", newline="\n")
    return r


def main():
    parser = argparse.ArgumentParser(description="Relatório e figuras da Etapa 5 (classes finais).")
    parser.add_argument("--rodada-final", type=int, required=True)
    parser.add_argument("--base-lista", type=int, choices=[0, 1], default=1,
                        help="base dos índices em indices_positivos.txt (padrão: 1)")
    args = parser.parse_args()
    try:
        r = executar(args.rodada_final, args.base_lista)
    except (ErroConsistencia, FileNotFoundError, RuntimeError) as erro:
        sys.exit(f"ERRO: {erro}")
    print(f"Morcegos: {len(r['morcegos'])}; sem rótulo (modelo): {len(r['sem_rotulo'])} {r['distrib']}")
    print(r["tab_fontes"].to_string())
    for o in r["segunda"].values():
        print(f"{o['nome']}: {o['n_acima']} sem rótulo >= {o['limiar']:.4g}; "
              f"interseção com o top {N_TOP} do k-NN: {len(o['intersecao'])}")
    print(f"UMAP: {r['umap_motivo']}")
    print(f"Relatório em results/{ARQ_RELATORIO}")


if __name__ == "__main__":
    main()
