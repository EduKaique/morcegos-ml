# Relatório da Etapa 3 — enriquecimento iterativo

Índices na base 0. Rodada 0 = `data/seeds/semente_reduzida.csv`.

## Configuração

- Classificador fixo (`src/config_modelo.py`, decisão da equipe a partir de `results/relatorio_etapa2.md`): `knn_frac`, {'n_neighbors': 3, 'weights': 'distance', 'metric': 'manhattan'}, sem escala.
- Limiar de alta confiança: 0.6728. Incertos: até 30 com 0 < pontuação < limiar, mais próximos do limiar primeiro (pontuação 0 não informa nada).
- Revisão manual: até 100 candidatos sem rótulo por rodada, por pontuação.
- A base só cresce com rótulos verificados: `gabarito` (grupo oculto, verificação automática) ou `humano` (decisão registrada em `revisao_manual.csv`). Teste e gêneros bloqueados são verificados antes e depois de cada rodada.
- Convergência: rodada sem positivo novo confirmado e sem pendências (no modo automático, as pendências sem rótulo não são resolvíveis e são só registradas).
- Precisão alta = morcegos / candidatos de alta confiança verificados pelo gabarito na rodada. Recall do oculto calculado com `src.validacao.avaliar_candidatos` sobre a lista acumulada de índices do oculto verificados.

## Modo automático (simulação, só gabarito)

Pasta: `data/rounds/auto`. Positivos e negativos na base ao fim de cada rodada; "pos. novos" inclui decisões humanas incorporadas.

| rodada | positivos | negativos | alta | incertos | pos. novos | neg. difíceis | FP alta | precisão alta | oculto encontrados | recall oculto | pendentes | convergido |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 73 | 541 | 56 | 12 | 60 | 5 | 0 | 1.000 | 60 | 0.984 | 3 | False |
| 2 | 74 | 546 | 1 | 8 | 1 | 5 | 0 | 1.000 | 61 | 1.000 | 3 | False |
| 3 | 74 | 546 | 0 | 3 | 0 | 0 | 0 | - | 61 | 1.000 | 3 | True |

**Motivo de parada / estado:** nenhum positivo novo confirmado; 3 candidato(s) sem rótulo ficaram pendentes (não resolvíveis no modo automático).

### Recuperação do oculto por família

| familia | n | encontrados | recall |
|---|---|---|---|
| Hipposideridae* | 1 | 1 | 100% |
| Molossidae* | 2 | 2 | 100% |
| Mormoopidae* | 3 | 3 | 100% |
| Mystacinidae* | 1 | 1 | 100% |
| Phyllostomidae | 7 | 7 | 100% |
| Pteropodidae | 2 | 2 | 100% |
| Rhinolophidae* | 10 | 10 | 100% |
| Vespertilionidae | 35 | 35 | 100% |

\* família sem representante na semente.

### Recuperação do oculto por gênero

| familia | genero | n | encontrados | recall |
|---|---|---|---|---|
| Hipposideridae | Hipposideros | 1 | 1 | 100% |
| Molossidae | Tadarida | 2 | 2 | 100% |
| Mormoopidae | Pteronotus | 3 | 3 | 100% |
| Mystacinidae | Mystacina | 1 | 1 | 100% |
| Phyllostomidae | Artibeus | 2 | 2 | 100% |
| Phyllostomidae | Brachyphylla | 1 | 1 | 100% |
| Phyllostomidae | Lophostoma | 1 | 1 | 100% |
| Phyllostomidae | Micronycteris | 1 | 1 | 100% |
| Phyllostomidae | Sturnira | 1 | 1 | 100% |
| Phyllostomidae | Vampyrum | 1 | 1 | 100% |
| Pteropodidae | Cynopterus | 1 | 1 | 100% |
| Pteropodidae | Epomophorus | 1 | 1 | 100% |
| Rhinolophidae | Rhinolophus | 10 | 10 | 100% |
| Vespertilionidae | Eptesicus | 1 | 1 | 100% |
| Vespertilionidae | Murina | 2 | 2 | 100% |
| Vespertilionidae | Myotis | 28 | 28 | 100% |
| Vespertilionidae | Pipistrellus | 2 | 2 | 100% |
| Vespertilionidae | Plecotus | 2 | 2 | 100% |

### Pendências (candidatos sem rótulo, última rodada)

3 candidato(s) sem rótulo; os 10 de maior pontuação:

| indice | nome | pontuacao | tipo | vizinhos (classe) |
|---|---|---|---|---|
| 4412 | Mammuthus_primigenius | 0.330022 | incerto | Choloepus_didactylus (0); Desmodus_rotundus (1); Ochotona_princeps (0) |
| 3311 | Helicosporidium_sp_ex_Simulium_jonesi | 0.329196 | incerto | Chlorella_sorokiniana (0); Coccomyxa_subellipsoidea_C_169 (0); Pipistrellus_abramus (1) |
| 4411 | Mammuthus_columbi | 0.322713 | incerto | Ochotona_princeps (0); Choloepus_didactylus (0); Desmodus_rotundus (1) |

## Rodadas manuais

Pasta: `data/rounds`. Positivos e negativos na base ao fim de cada rodada; "pos. novos" inclui decisões humanas incorporadas.

| rodada | positivos | negativos | alta | incertos | pos. novos | neg. difíceis | FP alta | precisão alta | oculto encontrados | recall oculto | pendentes | convergido |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 73 | 541 | 56 | 12 | 60 | 5 | 0 | 1.000 | 60 | 0.984 | 3 | False |
| 2 | 74 | 549 | 1 | 6 | 1 | 5 | 0 | 1.000 | 61 | 1.000 | 1 | False |
| 3 | 74 | 550 | 0 | 0 | 0 | 0 | 0 | - | 61 | 1.000 | 0 | True |

**Motivo de parada / estado:** nenhum positivo novo confirmado e nenhuma pendência de revisão.

### Recuperação do oculto por família

| familia | n | encontrados | recall |
|---|---|---|---|
| Hipposideridae* | 1 | 1 | 100% |
| Molossidae* | 2 | 2 | 100% |
| Mormoopidae* | 3 | 3 | 100% |
| Mystacinidae* | 1 | 1 | 100% |
| Phyllostomidae | 7 | 7 | 100% |
| Pteropodidae | 2 | 2 | 100% |
| Rhinolophidae* | 10 | 10 | 100% |
| Vespertilionidae | 35 | 35 | 100% |

\* família sem representante na semente.

### Recuperação do oculto por gênero

| familia | genero | n | encontrados | recall |
|---|---|---|---|---|
| Hipposideridae | Hipposideros | 1 | 1 | 100% |
| Molossidae | Tadarida | 2 | 2 | 100% |
| Mormoopidae | Pteronotus | 3 | 3 | 100% |
| Mystacinidae | Mystacina | 1 | 1 | 100% |
| Phyllostomidae | Artibeus | 2 | 2 | 100% |
| Phyllostomidae | Brachyphylla | 1 | 1 | 100% |
| Phyllostomidae | Lophostoma | 1 | 1 | 100% |
| Phyllostomidae | Micronycteris | 1 | 1 | 100% |
| Phyllostomidae | Sturnira | 1 | 1 | 100% |
| Phyllostomidae | Vampyrum | 1 | 1 | 100% |
| Pteropodidae | Cynopterus | 1 | 1 | 100% |
| Pteropodidae | Epomophorus | 1 | 1 | 100% |
| Rhinolophidae | Rhinolophus | 10 | 10 | 100% |
| Vespertilionidae | Eptesicus | 1 | 1 | 100% |
| Vespertilionidae | Murina | 2 | 2 | 100% |
| Vespertilionidae | Myotis | 28 | 28 | 100% |
| Vespertilionidae | Pipistrellus | 2 | 2 | 100% |
| Vespertilionidae | Plecotus | 2 | 2 | 100% |

### Pendências (candidatos sem rótulo, última rodada)

0 candidato(s) sem rótulo; os 10 de maior pontuação:

_(nenhum)_

## Limites

- O gabarito faz o papel do especialista no modo automático: é uma simulação otimista (verificação instantânea e sem erro).
- A pontuação do k-NN com k=3 é discreta; quase todo o pool tem pontuação 0, então há poucos incertos e o active learning tem pouco material.
- O oculto concentra os morcegos em *Myotis* (28) e *Rhinolophus* (10): o recall agregado muda aos saltos com esses dois gêneros.
- Os negativos prováveis da semente não foram verificados taxonomicamente.
- Candidatos sem rótulo não entram na base no modo automático; morcegos fora da lista taxonômica só seriam incorporados por decisão humana.
- O teste congelado não é usado aqui (Etapa 4).

## Figuras

- `results/figuras/etapa3_positivos_por_rodada.png`
- `results/figuras/etapa3_recall_familia_por_rodada.png`
- `results/figuras/etapa3_umap.png`

## Execução

Tempo do último comando (rodadas, projeção e figuras): 42.9 s.
