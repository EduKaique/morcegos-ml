"""Comparação de classificadores (Etapa 2): k-NN, SVM RBF e Random Forest.

Treina **só** com ``data/seeds/semente_reduzida.csv``. A seleção usa validação
agrupada por gênero (``validacao_agrupada``) repetida com as sementes 42..51,
com as **mesmas dobras** para todas as configurações. O grupo ``oculto`` do
gabarito é só diagnóstico (não escolhe hiperparâmetro nem modelo) e o teste
congelado não é tocado.

Uso: ``python -m src.experimento_etapa2`` a partir da raiz do repositório.
"""

import hashlib
import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # roda sem janela gráfica

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.neighbors import NearestNeighbors

from src.carregar_dados import DIR_RAIZ, carregar_tudo
from src.modelos import (COLUNAS, GRADE_PADRAO, MODELOS, construir_pipeline,
                         gerar_configuracoes, pontuar)
from src.supervisao import _tabela_md, instancias_bloqueadas
from src.validacao import validacao_agrupada

RANDOM_STATE = 42
SEMENTES_REPETICAO = tuple(range(42, 52))
N_SPLITS = 5
PRECISAO_ALVO = 0.95
N_VIZINHOS_DIAGNOSTICO = 3

DIR_SEEDS = DIR_RAIZ / "data" / "seeds"
DIR_RESULTADOS = DIR_RAIZ / "results"

NOMES_MODELOS = {"knn": "k-NN", "svm": "SVM RBF", "rf": "Random Forest"}
# Regra de desempate sugerida (simplicidade e interpretabilidade), da mais simples à menos.
ORDEM_SIMPLICIDADE = ("knn", "rf", "svm")

ARQUIVOS_SAIDA = ("etapa2_comparacao.csv", "etapa2_oculto_por_familia.csv",
                  "etapa2_oculto_vizinhos.csv", "etapa2_importancias_rf.csv",
                  "relatorio_etapa2.md")
FIGURAS_SAIDA = ("etapa2_curvas_pr.png", "etapa2_boxplot_ap.png",
                 "etapa2_recall_familia_oculto.png")


# --------------------------------------------------------------------------
# Entradas e proteção contra vazamento
# --------------------------------------------------------------------------

def _ler(caminho):
    return pd.read_csv(caminho, keep_default_na=False)


def carregar_entradas(dir_seeds=DIR_SEEDS):
    """Lê semente, gabarito, teste e gêneros bloqueados; devolve um dict com ``X``.

    ``bloqueados_idx`` são todas as instâncias do ``Names.txt`` dos gêneros do teste.
    """
    dir_seeds = Path(dir_seeds)
    df, X = carregar_tudo()
    semente = _ler(dir_seeds / "semente_reduzida.csv").reset_index(drop=True)
    semente["rotulo"] = semente["rotulo"].astype(int)
    gabarito = _ler(dir_seeds / "gabarito.csv")
    gabarito["rotulo_gabarito"] = gabarito["rotulo_gabarito"].astype(int)
    teste = _ler(dir_seeds / "teste_verificado.csv")
    arq_bloqueados = dir_seeds / "generos_bloqueados.csv"
    return {"df": df, "X": X, "semente": semente, "gabarito": gabarito, "teste": teste,
            "generos_bloqueados": set(_ler(arq_bloqueados)["genero_atual"]),
            "bloqueados_idx": set(instancias_bloqueadas(df, arq_bloqueados))}


def verificar_sem_vazamento(semente, teste_idx, bloqueados_idx, oculto_idx, generos_bloqueados=()):
    """Levanta ``ValueError`` se a semente tiver índice do teste, bloqueado ou oculto,
    ou algum gênero bloqueado."""
    usados = set(semente["indice"].astype(int))
    problemas = {
        "teste verificado": usados & set(teste_idx),
        "gêneros bloqueados": usados & set(bloqueados_idx),
        "grupo oculto": usados & set(oculto_idx),
    }
    for nome, inter in problemas.items():
        if inter:
            raise ValueError(f"Vazamento: {len(inter)} índice(s) do {nome} na semente: "
                             f"{sorted(inter)[:10]}")
    generos = set(semente["genero_atual"]) & set(generos_bloqueados)
    if generos:
        raise ValueError(f"Vazamento: gêneros bloqueados na semente: {sorted(generos)[:10]}")


# --------------------------------------------------------------------------
# Dobras, avaliação e métricas
# --------------------------------------------------------------------------

def gerar_dobras(semente, seeds=SEMENTES_REPETICAO, n_splits=N_SPLITS):
    """``{seed: [(idx_treino, idx_validacao), ...]}``, calculado uma vez para todos os modelos."""
    return {s: validacao_agrupada(semente, n_splits=n_splits, seed=s) for s in seeds}


def hash_dobras(dobras):
    """SHA256 das posições de validação de todas as dobras de todas as repetições."""
    h = hashlib.sha256()
    for seed in sorted(dobras):
        h.update(f"seed={seed};".encode())
        for _, idx_val in dobras[seed]:
            h.update(np.asarray(sorted(idx_val), dtype=np.int64).tobytes())
            h.update(b"|")
    return h.hexdigest()


def avaliar_configuracao(config, X, y, dobras, devolver_modelos=False):
    """Pontuações fora da amostra por repetição: ``{seed: array(len(y))}``.

    Cada dobra ajusta um pipeline novo só com ``X[idx_treino]``. Com
    ``devolver_modelos=True`` devolve também ``{seed: [(pipe, idx_treino), ...]}``.
    """
    base = construir_pipeline(config["variante"], config["params"], config["escala"])
    pontuacoes, modelos = {}, {}
    for seed, divisoes in dobras.items():
        s = np.full(len(y), np.nan)
        modelos[seed] = []
        for idx_tr, idx_val in divisoes:
            pipe = clone(base).fit(X[idx_tr], y[idx_tr])
            s[idx_val] = pontuar(pipe, X[idx_val])
            if devolver_modelos:
                modelos[seed].append((pipe, idx_tr))
        pontuacoes[seed] = s
    return (pontuacoes, modelos) if devolver_modelos else pontuacoes


def metricas_repeticao(y, s, alvo=PRECISAO_ALVO):
    """Métricas de uma repetição a partir das pontuações fora da amostra.

    ``limiar_p95`` é o menor limiar com precisão >= ``alvo`` (NaN se nenhum);
    ``recall_p95`` é o recall nesse limiar (o maior recall com precisão >= alvo).
    Sensibilidade, precisão e acurácia são medidas nesse limiar; se nenhum limiar
    atinge o alvo, usa o limiar do F1 máximo (``sem_limiar_p95 = 1``).
    """
    y = np.asarray(y).astype(int)
    prec, rec, lim = precision_recall_curve(y, s)
    p, r = prec[:-1], rec[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(p + r > 0, 2 * p * r / np.where(p + r > 0, p + r, 1), 0.0)
    i_f1 = int(np.argmax(f1))
    atinge = np.flatnonzero(p >= alvo)
    if len(atinge):
        i, sem = int(atinge[0]), 0
        limiar_p95, recall_p95 = float(lim[i]), float(r[i])
    else:
        i, sem = i_f1, 1
        limiar_p95, recall_p95 = np.nan, 0.0
    pred = s >= lim[i]
    vp = int((pred & (y == 1)).sum())
    return {
        "ap": float(average_precision_score(y, s)),
        "f1max": float(f1[i_f1]),
        "limiar_f1": float(lim[i_f1]),
        "recall_p95": recall_p95,
        "limiar_p95": limiar_p95,
        "sens": vp / max(int((y == 1).sum()), 1),
        "prec": vp / max(int(pred.sum()), 1),
        "acc": float((pred == (y == 1)).mean()),
        "sem_limiar_p95": sem,
    }


def resumir(config, por_rep, hash_):
    """Uma linha do CSV comparativo: média e desvio entre repetições."""
    tab = pd.DataFrame(por_rep)
    linha = {"modelo": config["modelo"], "variante": config["variante"],
             "escala": config["escala"], "config_id": config["config_id"],
             "params": json.dumps(config["params"], sort_keys=True)}
    for m in ("ap", "f1max", "recall_p95", "sens", "prec", "acc"):
        linha[f"{m}_media"] = tab[m].mean()
        linha[f"{m}_dp"] = tab[m].std()
    lim = tab["limiar_p95"].dropna()
    linha["limiar_p95_mediana"] = lim.median() if len(lim) else np.nan
    linha["limiar_p95_media"] = lim.mean() if len(lim) else np.nan
    linha["limiar_p95_dp"] = lim.std() if len(lim) > 1 else np.nan
    linha["limiar_f1_mediana"] = tab["limiar_f1"].median()
    linha["n_rep_sem_limiar_p95"] = int(tab["sem_limiar_p95"].sum())
    linha["n_repeticoes"] = len(tab)
    linha["hash_dobras"] = hash_
    return linha


# --------------------------------------------------------------------------
# Seleção e comparação
# --------------------------------------------------------------------------

def melhores(comparacao, por=("modelo",)):
    """Melhor configuração por grupo: maior ``ap_media``; desempate pelo menor ``ap_dp``."""
    ordenada = comparacao.sort_values(["ap_media", "ap_dp"], ascending=[False, True], kind="stable")
    return ordenada.groupby(list(por), sort=False).head(1).set_index(list(por))


def comparar_modelos(melhor):
    """Tabela dos melhores de cada modelo com a diferença para o vencedor e empate técnico."""
    tab = melhor.loc[[m for m in MODELOS if m in melhor.index]].copy()
    vencedor = tab["ap_media"].idxmax()
    tab["delta_ap"] = tab.loc[vencedor, "ap_media"] - tab["ap_media"]
    tab["empate_tecnico"] = [
        m != vencedor and d < max(tab.loc[vencedor, "ap_dp"], tab.loc[m, "ap_dp"])
        for m, d in tab["delta_ap"].items()]
    return tab, vencedor


def sugestao_desempate(tab, vencedor):
    """Modelo sugerido pela regra: entre o vencedor e os empatados, o mais simples."""
    candidatos = [vencedor] + list(tab.index[tab["empate_tecnico"]])
    return min(candidatos, key=ORDEM_SIMPLICIDADE.index)


def limiar_sugerido(linha):
    """Mediana dos limiares com precisão >= 0,95; sem eles, mediana do limiar do F1 máximo."""
    if pd.notna(linha["limiar_p95_mediana"]):
        return float(linha["limiar_p95_mediana"]), "precisao>=0.95"
    return float(linha["limiar_f1_mediana"]), "F1 maximo (nenhum limiar atingiu 0.95)"


# --------------------------------------------------------------------------
# Diagnóstico no oculto (não influencia a escolha)
# --------------------------------------------------------------------------

def _config_de(linha):
    return {"variante": linha["variante"], "escala": bool(linha["escala"]),
            "params": json.loads(linha["params"])}


def ajustar_na_semente(linha, X_sem, y_sem):
    c = _config_de(linha)
    return construir_pipeline(c["variante"], c["params"], c["escala"]).fit(X_sem, y_sem)


def diagnostico_oculto(melhor, limiares, X, semente, gabarito):
    """Pontua o oculto com o melhor de cada modelo treinado na semente inteira.

    Devolve ``(pontuacoes, por_familia, falsos_positivos, ap_oculto, pipes)``.
    """
    X_sem, y_sem = X[semente["indice"]], semente["rotulo"].to_numpy()
    oculto = gabarito[gabarito["grupo"] == "oculto"].reset_index(drop=True)
    familias_semente = set(semente.loc[semente["rotulo"] == 1, "familia"])
    pos = oculto[oculto["rotulo_gabarito"] == 1]
    linhas, fps, ap_oculto, pipes = [], [], {}, {}
    pontuacoes = oculto[["indice", "nome", "genero_atual", "ordem", "familia", "rotulo_gabarito"]].copy()
    for modelo in MODELOS:
        pipe = ajustar_na_semente(melhor.loc[modelo], X_sem, y_sem)
        pipes[modelo] = pipe
        s = pontuar(pipe, X[oculto["indice"]])
        achou = s >= limiares[modelo]
        pontuacoes[f"pontuacao_{modelo}"] = s
        pontuacoes[f"encontrado_{modelo}"] = achou
        ap_oculto[modelo] = float(average_precision_score(oculto["rotulo_gabarito"], s))
        sub = pos.assign(encontrado=achou[pos.index])
        for nivel, chaves in (("familia", ["familia"]), ("genero", ["familia", "genero_atual"])):
            for valores, g in sub.groupby(chaves, sort=True):
                valores = valores if isinstance(valores, tuple) else (valores,)
                fam = valores[0]
                linhas.append({"modelo": modelo, "nivel": nivel, "familia": fam,
                               "genero": valores[1] if nivel == "genero" else "",
                               "n_morcegos": len(g), "encontrados": int(g["encontrado"].sum()),
                               "recall": g["encontrado"].mean(),
                               "familia_na_semente": fam in familias_semente})
        neg = oculto["rotulo_gabarito"].to_numpy() == 0
        for i in np.flatnonzero(neg & achou):
            fps.append({"modelo": modelo, "indice": int(oculto.at[i, "indice"]),
                        "nome": oculto.at[i, "nome"], "ordem": oculto.at[i, "ordem"],
                        "familia": oculto.at[i, "familia"], "pontuacao": round(float(s[i]), 4)})
    falsos = pd.DataFrame(fps, columns=["modelo", "indice", "nome", "ordem", "familia", "pontuacao"])
    return pontuacoes, pd.DataFrame(linhas), falsos, ap_oculto, pipes


def vizinhos_knn(linha_knn, pipe, X, semente, pontuacoes, k=N_VIZINHOS_DIAGNOSTICO):
    """Para cada morcego do oculto, os ``k`` vizinhos mais próximos na semente."""
    escala = pipe.named_steps["escala"]
    transformar = (lambda Z: Z) if escala == "passthrough" else escala.transform
    metrica = json.loads(linha_knn["params"])["metric"]
    nn = NearestNeighbors(n_neighbors=k, metric=metrica, algorithm="brute")
    nn.fit(transformar(X[semente["indice"]]))
    morcegos = pontuacoes[pontuacoes["rotulo_gabarito"] == 1].reset_index(drop=True)
    dist, pos = nn.kneighbors(transformar(X[morcegos["indice"]]))
    linhas = []
    for i, m in morcegos.iterrows():
        linha = {"indice": int(m["indice"]), "nome": m["nome"], "genero": m["genero_atual"],
                 "familia": m["familia"], "pontuacao": round(float(m["pontuacao_knn"]), 4),
                 "resultado": "encontrado" if m["encontrado_knn"] else "perdido"}
        for j in range(k):
            v = semente.iloc[pos[i, j]]
            linha[f"viz{j + 1}_nome"] = v["nome"]
            linha[f"viz{j + 1}_classe"] = int(v["rotulo"])
            linha[f"viz{j + 1}_dist"] = round(float(dist[i, j]), 4)
        linhas.append(linha)
    return pd.DataFrame(linhas)


def importancias_rf(linha_rf, X, semente, dobras, pipe_semente):
    """Importâncias da melhor RF na semente inteira e média/desvio nas dobras."""
    X_sem, y_sem = X[semente["indice"]], semente["rotulo"].to_numpy()
    _, modelos = avaliar_configuracao(_config_de(linha_rf), X_sem, y_sem, dobras,
                                      devolver_modelos=True)
    cv = np.array([pipe[-1].feature_importances_ for lst in modelos.values() for pipe, _ in lst])
    tab = pd.DataFrame({"caracteristica": COLUNAS[:X.shape[1]],
                        "importancia_semente": pipe_semente[-1].feature_importances_,
                        "importancia_cv_media": cv.mean(axis=0),
                        "importancia_cv_dp": cv.std(axis=0, ddof=1) if len(cv) > 1 else np.nan})
    tab = tab.sort_values("importancia_semente", ascending=False, kind="stable")
    tab["posicao"] = np.arange(1, len(tab) + 1)
    return tab


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------

def figura_curvas_pr(melhor_escala, oof, y, seed, caminho):
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for eixo, escala in zip(eixos, (False, True)):
        for modelo in MODELOS:
            if (modelo, escala) not in melhor_escala.index:
                continue
            linha = melhor_escala.loc[(modelo, escala)]
            prec, rec, _ = precision_recall_curve(y, oof[linha["config_id"]][seed])
            eixo.step(rec, prec, where="post",
                      label=f"{NOMES_MODELOS[modelo]} (AP {linha['ap_media']:.2f} ± {linha['ap_dp']:.2f})")
        eixo.axhline(PRECISAO_ALVO, color="gray", ls=":", lw=1)
        eixo.set_title("com StandardScaler" if escala else "sem escala")
        eixo.set_xlabel("recall")
        eixo.legend(fontsize=8, loc="lower left")
    eixos[0].set_ylabel("precisão")
    fig.suptitle(f"Curvas precisão-recall (validação agrupada, repetição seed={seed})")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def figura_boxplot_ap(melhor_escala, ap_por_rep, caminho):
    dados, rotulos = [], []
    for modelo in MODELOS:
        for escala in (False, True):
            if (modelo, escala) in melhor_escala.index:
                dados.append(ap_por_rep[melhor_escala.loc[(modelo, escala), "config_id"]])
                rotulos.append(f"{NOMES_MODELOS[modelo]}\n{'com' if escala else 'sem'} escala")
    fig, eixo = plt.subplots(figsize=(9, 4.5))
    eixo.boxplot(dados)
    eixo.set_xticks(range(1, len(rotulos) + 1), rotulos, fontsize=8)
    eixo.set_ylabel("precisão média (AP)")
    eixo.set_title("AP por repetição: melhor configuração de cada modelo")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def figura_recall_familia(por_familia, caminho):
    fam = por_familia[por_familia["nivel"] == "familia"]
    familias = sorted(fam["familia"].unique())
    na_semente = fam.groupby("familia")["familia_na_semente"].first()
    largura = 0.8 / len(MODELOS)
    fig, eixo = plt.subplots(figsize=(10, 4.5))
    for j, modelo in enumerate(MODELOS):
        rec = fam[fam["modelo"] == modelo].set_index("familia")["recall"].reindex(familias)
        eixo.bar(np.arange(len(familias)) + (j - 1) * largura, rec.fillna(0), largura,
                 label=NOMES_MODELOS[modelo])
    n = fam.groupby("familia")["n_morcegos"].first()
    rotulos = [f"{f}{'' if na_semente[f] else '*'}\n(n={n[f]})" for f in familias]
    eixo.set_xticks(np.arange(len(familias)), rotulos, fontsize=7, rotation=30, ha="right")
    eixo.set_ylim(0, 1.05)
    eixo.set_ylabel("recall no oculto")
    eixo.set_title("Recall por família no oculto (* = família sem representante na semente)")
    eixo.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Relatório
# --------------------------------------------------------------------------

def _mdp(linha, m, casas=3):
    return f"{linha[f'{m}_media']:.{casas}f} ± {linha[f'{m}_dp']:.{casas}f}"


def gerar_relatorio(ctx):
    semente, melhor_escala, tab = ctx["semente"], ctx["melhor_escala"], ctx["tabela_modelos"]
    vencedor, sugerido = ctx["vencedor"], ctx["sugerido"]
    n_pos = int(semente["rotulo"].sum())
    n_gen_pos = semente.loc[semente["rotulo"] == 1, "genero_atual"].nunique()
    origem_neg = semente.loc[semente["rotulo"] == 0, "origem"]
    n_neg = len(origem_neg)
    n_prov = int((origem_neg == "negativo_provavel").sum())
    n_mam = n_neg - n_prov
    seeds = ctx["seeds"]
    linhas = [
        "# Relatório da Etapa 2 — comparação de classificadores", "",
        "Índices na base 0. Treino **somente** com `data/seeds/semente_reduzida.csv` "
        f"({len(semente)} instâncias: {n_pos} morcegos de {n_gen_pos} gêneros, "
        f"{len(semente) - n_pos} negativos). O oculto é só diagnóstico; o teste congelado "
        "e os gêneros bloqueados não foram usados (verificado antes de qualquer ajuste).", "",
        "## Desenho", "",
        f"- Validação agrupada por gênero (`StratifiedGroupKFold`, {ctx['n_splits']} dobras), "
        f"repetida {len(seeds)} vezes (sementes {seeds[0]}..{seeds[-1]}). As mesmas dobras "
        f"para todas as {ctx['n_configs']} configurações (hash `{ctx['hash'][:12]}`).",
        "- Em cada repetição, as pontuações fora da amostra das dobras são juntadas e as "
        "métricas são calculadas sobre todas as instâncias da semente; a tabela mostra média ± "
        "desvio-padrão entre repetições.",
        "- Pontuação (maior = mais provável morcego): k-NN fração ponderada de vizinhos "
        "positivos (`knn_frac`) ou razão d_neg/(d_pos+d_neg) (`knn_razao`); SVM "
        "`decision_function`; RF `predict_proba`.",
        f"- Limiar escolhido = menor limiar com precisão ≥ {PRECISAO_ALVO} na validação "
        "agrupada; sensibilidade, precisão e acurácia medidas nele (sem esse limiar, usa o do F1 "
        "máximo). Limiar sugerido = mediana entre repetições.",
        f"- Todas as configurações e métricas: `results/etapa2_comparacao.csv`.", "",
        "## Melhor configuração de cada modelo, sem e com escala", "",
    ]
    t = []
    for modelo in MODELOS:
        for escala in (False, True):
            if (modelo, escala) not in melhor_escala.index:
                continue
            r = melhor_escala.loc[(modelo, escala)]
            t.append({"modelo": NOMES_MODELOS[modelo], "escala": "com" if escala else "sem",
                      "configuracao": f"`{r['config_id']}`", "AP": _mdp(r, "ap"),
                      "F1 max": _mdp(r, "f1max"), "recall@P>=0.95": _mdp(r, "recall_p95"),
                      "sens": _mdp(r, "sens"), "prec": _mdp(r, "prec"), "acc": _mdp(r, "acc"),
                      "reps sem P>=0.95": int(r["n_rep_sem_limiar_p95"])})
    linhas += [_tabela_md(pd.DataFrame(t)), "## Comparação entre os três modelos", ""]
    t = []
    for modelo, r in tab.iterrows():
        lim, origem = ctx["limiares"][modelo]
        t.append({"modelo": NOMES_MODELOS[modelo], "melhor configuracao": f"`{r['config_id']}`",
                  "AP": _mdp(r, "ap"), "diferenca para o vencedor": f"{r['delta_ap']:.3f}",
                  "empate tecnico (< 1 dp)": "vencedor" if modelo == vencedor
                  else ("sim" if r["empate_tecnico"] else "não"),
                  "limiar sugerido": f"{lim:.4g}", "origem do limiar": origem,
                  "limiar P>=0.95 (media ± dp)":
                      f"{r['limiar_p95_media']:.4g} ± {r['limiar_p95_dp']:.4g}"
                      if pd.notna(r["limiar_p95_media"]) else "-"})
    linhas += [_tabela_md(pd.DataFrame(t)),
               f"**Maior AP:** {NOMES_MODELOS[vencedor]}. Empate técnico = diferença de AP menor "
               "que o maior dos dois desvios-padrão.", "",
               "Configurações empatadas com a maior AP de cada modelo (a escolhida é a primeira "
               "na ordem da grade): " + ", ".join(
                   f"{NOMES_MODELOS[m]} {n}" for m, n in ctx["empates_no_topo"].items()) + ".", "",
               "**Regra de desempate sugerida:** se houver empate técnico, preferir o modelo mais "
               "simples e interpretável, na ordem k-NN (decisão explicável pelos vizinhos, que têm "
               "nome) → Random Forest (importância das características) → SVM RBF (fronteira no "
               f"espaço do kernel). Por essa regra a sugestão seria **{NOMES_MODELOS[sugerido]}**. "
               "**A decisão final é da equipe.**", "",
               "**Limiares:** o SVM usa `decision_function`, cuja escala muda a cada ajuste; o "
               "limiar transfere para o modelo treinado na semente inteira só de forma aproximada. "
               "A pontuação do k-NN por fração é discreta (poucos valores possíveis com k pequeno).",
               "", "## Diagnóstico no oculto (não usado na escolha)", "",
               "Melhor configuração de cada modelo treinada na semente inteira; \"encontrado\" = "
               "pontuação ≥ limiar sugerido do modelo. Famílias marcadas com * não têm morcego na "
               "semente.", ""]
    ap_oc = ctx["ap_oculto"]
    linhas += [_tabela_md(pd.DataFrame([{"modelo": NOMES_MODELOS[m], "AP no oculto": f"{ap_oc[m]:.3f}",
                                         "morcegos encontrados": f"{int(ctx['pontuacoes'][f'encontrado_{m}'][ctx['pontuacoes']['rotulo_gabarito'] == 1].sum())}/"
                                                                 f"{int((ctx['pontuacoes']['rotulo_gabarito'] == 1).sum())}",
                                         "falsos positivos (mamíferos)": int((ctx['falsos']['modelo'] == m).sum())}
                                        for m in MODELOS])), "### Recall por família", ""]
    pf = ctx["por_familia"]
    for nivel, coluna in (("familia", "familia"), ("genero", "genero")):
        sub = pf[pf["nivel"] == nivel]
        chave = ["familia", "genero"] if nivel == "genero" else ["familia"]
        largo = sub.pivot_table(index=chave, columns="modelo", values="encontrados", aggfunc="first")
        n = sub.groupby(chave)["n_morcegos"].first()
        na = sub.groupby(chave)["familia_na_semente"].first()
        t = []
        for k in largo.index:
            k_t = k if isinstance(k, tuple) else (k,)
            linha = {"familia": k_t[0] + ("" if na[k] else "*")}
            if nivel == "genero":
                linha["genero"] = k_t[1]
            linha["n"] = int(n[k])
            for m in MODELOS:
                linha[NOMES_MODELOS[m]] = f"{int(largo.loc[k, m])} ({largo.loc[k, m] / n[k]:.0%})"
            t.append(linha)
        if nivel == "genero":
            linhas += ["### Recall por gênero", ""]
        linhas += [_tabela_md(pd.DataFrame(t))]
    linhas += ["### Falsos positivos entre mamíferos não-morcegos do oculto", ""]
    fp = ctx["falsos"].assign(modelo=lambda d: d["modelo"].map(NOMES_MODELOS))
    linhas += [_tabela_md(fp), "### Vizinhos do k-NN", "",
               f"Configuração `{ctx['melhor'].loc['knn', 'config_id']}`; detalhes em "
               "`results/etapa2_oculto_vizinhos.csv` (3 vizinhos mais próximos na semente de cada "
               "morcego do oculto).", ""]
    viz = ctx["vizinhos"]
    if not viz.empty:
        viz = viz.assign(n_viz_morcego=viz[[c for c in viz.columns if c.endswith("_classe")]].sum(axis=1))
        resumo = (viz.groupby(["familia", "resultado"]).agg(morcegos=("indice", "size"),
                  media_vizinhos_morcego=("n_viz_morcego", "mean")).round(2).reset_index())
        linhas += [_tabela_md(resumo)]
    imp = ctx["importancias"].head(10)[["posicao", "caracteristica", "importancia_semente",
                                        "importancia_cv_media", "importancia_cv_dp"]].round(4)
    linhas += ["## Importância das características (melhor Random Forest)", "",
               f"Configuração `{ctx['melhor'].loc['rf', 'config_id']}`; 10 primeiras "
               "(todas em `results/etapa2_importancias_rf.csv`).", "", _tabela_md(imp),
               "## Limites", "",
               "- **Saturação:** várias configurações ficam com AP perto de 1 e as diferenças "
               "entre modelos são da ordem de 0,01, então a validação mal distingue os modelos. "
               "Explicação mais provável (não comprovada): os morcegos formam um grupo bem "
               "separado no espaço das 50 características e há poucos positivos. Dos "
               f"{n_neg} negativos da semente, os negativos prováveis ({n_prov}, "
               f"{n_prov / n_neg:.0%}) são fáceis de separar, mas os mamíferos não-morcegos "
               f"({n_mam}, {n_mam / n_neg:.0%}) também ficam bem separados. O comportamento nas "
               "famílias sem representante (diagnóstico do oculto) diz mais sobre a prospecção.",
               "- A Random Forest dá o mesmo resultado sem e com escala (árvores não dependem da "
               "escala das colunas), como esperado.",
               f"- Só {n_pos} positivos de {n_gen_pos} gêneros na semente: cada dobra valida com "
               "1 a 2 gêneros de morcego, então as métricas são ruidosas (ver desvios e boxplot).",
               "- Os negativos prováveis da semente não foram verificados taxonomicamente (ruído "
               "de rótulo possível).",
               "- O oculto concentra a maioria dos morcegos em *Myotis* (28) e *Rhinolophus* (10): "
               "acertar ou errar um desses gêneros muda o recall agregado de forma abrupta.",
               "- Rhinolophidae, Hipposideridae, Molossidae, Mormoopidae e Mystacinidae não têm "
               "representante na semente.",
               "- Precisão ≥ 0,95 com 13 positivos depende de pouquíssimos exemplos; o limiar é "
               "uma sugestão inicial para a Etapa 3, a ser revisto a cada rodada.",
               "- O oculto não escolhe nada aqui; usá-lo para escolher invalidaria a medida de "
               "prospecção da Etapa 3.", "",
               "## Figuras", "",
               "- `results/figuras/etapa2_curvas_pr.png`", "- `results/figuras/etapa2_boxplot_ap.png`",
               "- `results/figuras/etapa2_recall_familia_oculto.png`", "",
               "## Execução", "", f"Tempo total: {ctx['tempo_s']:.1f} s.", ""]
    return "\n".join(linhas)


# --------------------------------------------------------------------------
# Orquestração
# --------------------------------------------------------------------------

def executar(dir_seeds=DIR_SEEDS, dir_resultados=DIR_RESULTADOS, seeds=SEMENTES_REPETICAO,
             grade=GRADE_PADRAO, n_splits=N_SPLITS):
    """Executa a Etapa 2 de ponta a ponta e devolve as tabelas num dict."""
    inicio = time.perf_counter()
    seeds = tuple(seeds)
    ent = carregar_entradas(dir_seeds)
    X, semente, gabarito = ent["X"], ent["semente"], ent["gabarito"]
    oculto_idx = gabarito.loc[gabarito["grupo"] == "oculto", "indice"]
    verificar_sem_vazamento(semente, ent["teste"]["indice"], ent["bloqueados_idx"], oculto_idx,
                            ent["generos_bloqueados"])

    X_sem, y_sem = X[semente["indice"]], semente["rotulo"].to_numpy()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*n_splits reduzido.*")
        dobras = gerar_dobras(semente, seeds, n_splits)
    hash_ = hash_dobras(dobras)

    configs = gerar_configuracoes(grade)
    oof, ap_por_rep, linhas = {}, {}, []
    for config in configs:
        oof[config["config_id"]] = avaliar_configuracao(config, X_sem, y_sem, dobras)
        por_rep = [metricas_repeticao(y_sem, s) for s in oof[config["config_id"]].values()]
        ap_por_rep[config["config_id"]] = [r["ap"] for r in por_rep]
        linhas.append(resumir(config, por_rep, hash_))
    comparacao = pd.DataFrame(linhas)

    melhor = melhores(comparacao)
    melhor_escala = melhores(comparacao, ("modelo", "escala"))
    tab, vencedor = comparar_modelos(melhor)
    limiares = {m: limiar_sugerido(melhor.loc[m]) for m in MODELOS}
    pontuacoes, por_familia, falsos, ap_oculto, pipes = diagnostico_oculto(
        melhor, {m: v[0] for m, v in limiares.items()}, X, semente, gabarito)
    vizinhos = vizinhos_knn(melhor.loc["knn"], pipes["knn"], X, semente, pontuacoes)
    importancias = importancias_rf(melhor.loc["rf"], X, semente, dobras, pipes["rf"])

    dir_resultados = Path(dir_resultados)
    dir_figuras = dir_resultados / "figuras"
    dir_figuras.mkdir(parents=True, exist_ok=True)
    comparacao.to_csv(dir_resultados / "etapa2_comparacao.csv", index=False)
    por_familia.to_csv(dir_resultados / "etapa2_oculto_por_familia.csv", index=False)
    vizinhos.to_csv(dir_resultados / "etapa2_oculto_vizinhos.csv", index=False)
    importancias.to_csv(dir_resultados / "etapa2_importancias_rf.csv", index=False)
    figura_curvas_pr(melhor_escala, oof, y_sem, seeds[0], dir_figuras / "etapa2_curvas_pr.png")
    figura_boxplot_ap(melhor_escala, ap_por_rep, dir_figuras / "etapa2_boxplot_ap.png")
    figura_recall_familia(por_familia, dir_figuras / "etapa2_recall_familia_oculto.png")

    tempo = time.perf_counter() - inicio
    ctx = {"semente": semente, "melhor": melhor, "melhor_escala": melhor_escala,
           "tabela_modelos": tab, "vencedor": vencedor, "sugerido": sugestao_desempate(tab, vencedor),
           "limiares": limiares, "seeds": seeds, "n_splits": len(next(iter(dobras.values()))),
           "n_configs": len(configs), "hash": hash_,
           "empates_no_topo": {m: int((g["ap_media"] >= g["ap_media"].max() - 1e-12).sum())
                               for m, g in comparacao.groupby("modelo", sort=False)}, "ap_oculto": ap_oculto,
           "pontuacoes": pontuacoes, "por_familia": por_familia, "falsos": falsos,
           "vizinhos": vizinhos, "importancias": importancias, "tempo_s": tempo}
    (dir_resultados / "relatorio_etapa2.md").write_text(gerar_relatorio(ctx), encoding="utf-8")

    indices_usados = set(semente["indice"].astype(int))
    return {**ctx, "comparacao": comparacao, "dobras": dobras, "indices_usados": indices_usados,
            "entradas": ent}


def main():
    res = executar()
    print(f"Resultados em {DIR_RESULTADOS} (relatorio_etapa2.md, etapa2_*.csv, figuras/etapa2_*.png)")
    print(f"Maior AP: {NOMES_MODELOS[res['vencedor']]}; sugestão pela regra de desempate: "
          f"{NOMES_MODELOS[res['sugerido']]} (decisão final da equipe).")
    print(f"Tempo total: {res['tempo_s']:.1f} s")


if __name__ == "__main__":
    main()
