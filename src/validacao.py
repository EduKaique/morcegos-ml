"""Validação cruzada agrupada por gênero.

Espécies do mesmo gênero são quase duplicatas no espaço de características;
se um gênero aparecer em treino e validação ao mesmo tempo, o desempenho fica
inflado. Este módulo gera as divisões e mede a prospecção de candidatos
no pool oculto; os classificadores são a Etapa 2.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RANDOM_STATE = 42


def _coluna_grupo(df):
    """Usa ``genero_atual`` (gênero após sinônimos) se existir; senão ``genero``."""
    return "genero_atual" if "genero_atual" in df.columns else "genero"


def validacao_agrupada(df_treino, n_splits=5, seed=RANDOM_STATE):
    """Gera divisões treino/validação estratificadas por ``rotulo`` e agrupadas por gênero.

    Usa ``StratifiedGroupKFold``: nenhum gênero aparece em treino e validação
    na mesma dobra. Se houver menos gêneros positivos que ``n_splits``, reduz
    ``n_splits`` com aviso (cada dobra precisa de ao menos um gênero positivo);
    com menos de 2 gêneros positivos, levanta ``ValueError``.

    Retorna uma lista de pares ``(idx_treino, idx_validacao)`` com posições
    (``iloc``) em ``df_treino``.
    """
    grupos = df_treino[_coluna_grupo(df_treino)].to_numpy()
    y = df_treino["rotulo"].astype(int).to_numpy()
    n_generos_positivos = len(set(grupos[y == 1]))
    if n_generos_positivos < 2:
        raise ValueError(
            f"Só {n_generos_positivos} gênero(s) positivo(s); são necessários ao menos 2 para validar.")
    if n_generos_positivos < n_splits:
        warnings.warn(f"Apenas {n_generos_positivos} gêneros positivos: n_splits reduzido "
                      f"de {n_splits} para {n_generos_positivos}.")
        n_splits = n_generos_positivos
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, grupos))


def avaliar_candidatos(candidatos_idx, gabarito):
    """Métricas de prospecção dos candidatos adicionados, medidas no pool oculto.

    ``candidatos_idx``: índices (base 0) dos candidatos a positivo, na ordem em
    que foram adicionados. ``gabarito``: DataFrame de ``data/seeds/gabarito.csv``
    (colunas ``indice``, ``rotulo_gabarito`` e ``grupo``).

    - candidato do grupo ``teste`` → ``ValueError`` (vazamento de gênero);
    - índice repetido → ``ValueError``;
    - candidato do grupo ``semente`` já era conhecido: não conta no pool;
    - candidato fora do gabarito (sem rótulo taxonômico) não conta no pool,
      mas é somado em ``fora_do_pool_acum``.

    Retorna um DataFrame com uma linha por candidato: ``passo``, ``indice``,
    ``no_pool``, ``morcego_real``, ``morcegos_reais_acum``,
    ``candidatos_no_pool_acum``, ``precisao_acum`` (morcegos reais / candidatos
    no pool), ``recall_acum`` (morcegos reais / morcegos do pool oculto) e
    ``fora_do_pool_acum``. Não treina nenhum classificador.
    """
    candidatos = [int(i) for i in candidatos_idx]
    if len(set(candidatos)) != len(candidatos):
        raise ValueError("Há índices de candidatos repetidos.")
    grupo = dict(zip(gabarito["indice"].astype(int), gabarito["grupo"]))
    rotulo = dict(zip(gabarito["indice"].astype(int), gabarito["rotulo_gabarito"].astype(int)))
    vazados = [i for i in candidatos if grupo.get(i) == "teste"]
    if vazados:
        raise ValueError(f"Candidatos de gêneros do teste (vazamento): {vazados[:10]}")

    oculto = gabarito[gabarito["grupo"] == "oculto"]
    total_morcegos = int((oculto["rotulo_gabarito"].astype(int) == 1).sum())

    no_pool = np.array([grupo.get(i) == "oculto" for i in candidatos], dtype=bool)
    morcego = np.array([n and rotulo[i] == 1 for i, n in zip(candidatos, no_pool)], dtype=bool)
    fora = np.array([i not in grupo for i in candidatos], dtype=bool)
    acertos = np.cumsum(morcego)
    no_pool_acum = np.cumsum(no_pool)
    with np.errstate(invalid="ignore", divide="ignore"):
        precisao = np.where(no_pool_acum > 0, acertos / np.maximum(no_pool_acum, 1), np.nan)
    recall = acertos / total_morcegos if total_morcegos else np.full(len(candidatos), np.nan)

    return pd.DataFrame({
        "passo": np.arange(1, len(candidatos) + 1),
        "indice": candidatos,
        "no_pool": no_pool,
        "morcego_real": morcego,
        "morcegos_reais_acum": acertos,
        "candidatos_no_pool_acum": no_pool_acum,
        "precisao_acum": precisao,
        "recall_acum": recall,
        "fora_do_pool_acum": np.cumsum(fora),
    })
