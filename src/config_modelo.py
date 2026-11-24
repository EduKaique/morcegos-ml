"""Configuração fixa do classificador usado a partir da Etapa 3 (decisão da equipe).

Origem: ``results/relatorio_etapa2.md``. O k-NN ``knn_frac`` (k=3, votos
ponderados por distância, métrica manhattan, sem escala) teve a maior precisão
média na validação agrupada por gênero repetida (AP 1,000 ± 0,000), e pela
regra de desempate (simplicidade e interpretabilidade) é o modelo sugerido.
O limiar 0.6728 é a mediana, entre as 10 repetições, do menor limiar com
precisão >= 0,95 nessa validação. Na Etapa 3 ele é o padrão de ``--limiar``.
"""

from src.modelos import construir_pipeline

RANDOM_STATE = 42
VARIANTE = "knn_frac"
PARAMS = {"n_neighbors": 3, "weights": "distance", "metric": "manhattan"}
ESCALA = False
LIMIAR_PADRAO = 0.6728


def construir_modelo():
    """Pipeline do classificador escolhido (sem escala), ainda não ajustado."""
    return construir_pipeline(VARIANTE, PARAMS, ESCALA)
