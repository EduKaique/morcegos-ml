"""Validação cruzada agrupada por gênero.

Espécies do mesmo gênero são quase duplicatas no espaço de características;
se um gênero aparecer em treino e validação ao mesmo tempo, o desempenho fica
inflado. Este módulo só gera as divisões; os classificadores são a Etapa 2.
"""

import warnings

import numpy as np
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
