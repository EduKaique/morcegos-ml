# Relatório da Etapa 2 — comparação de classificadores

Índices na base 0. Treino **somente** com `data/seeds/semente_reduzida.csv` (549 instâncias: 13 morcegos de 9 gêneros, 536 negativos). O oculto é só diagnóstico; o teste congelado e os gêneros bloqueados não foram usados (verificado antes de qualquer ajuste).

## Desenho

- Validação agrupada por gênero (`StratifiedGroupKFold`, 5 dobras), repetida 10 vezes (sementes 42..51). As mesmas dobras para todas as 98 configurações (hash `8c30e9cb58de`).
- Em cada repetição, as pontuações fora da amostra das dobras são juntadas e as métricas são calculadas sobre todas as instâncias da semente; a tabela mostra média ± desvio-padrão entre repetições.
- Pontuação (maior = mais provável morcego): k-NN fração ponderada de vizinhos positivos (`knn_frac`) ou razão d_neg/(d_pos+d_neg) (`knn_razao`); SVM `decision_function`; RF `predict_proba`.
- Limiar escolhido = menor limiar com precisão ≥ 0.95 na validação agrupada; sensibilidade, precisão e acurácia medidas nele (sem esse limiar, usa o do F1 máximo). Limiar sugerido = mediana entre repetições.
- Todas as configurações e métricas: `results/etapa2_comparacao.csv`.

## Melhor configuração de cada modelo, sem e com escala

| modelo | escala | configuracao | AP | F1 max | recall@P>=0.95 | sens | prec | acc | reps sem P>=0.95 |
|---|---|---|---|---|---|---|---|---|---|
| k-NN | sem | `knn_frac|k=3|w=distance|m=manhattan|sem_escala` | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0 |
| k-NN | com | `knn_frac|k=5|w=distance|m=manhattan|escala` | 0.998 ± 0.004 | 0.992 ± 0.017 | 0.985 ± 0.032 | 0.985 ± 0.032 | 1.000 ± 0.000 | 1.000 ± 0.001 | 0 |
| SVM RBF | sem | `svm_rbf|C=10|gamma=1|sem_escala` | 0.997 ± 0.007 | 0.992 ± 0.017 | 0.985 ± 0.032 | 0.985 ± 0.032 | 1.000 ± 0.000 | 1.000 ± 0.001 | 0 |
| SVM RBF | com | `svm_rbf|C=10|gamma=scale|escala` | 0.991 ± 0.010 | 0.976 ± 0.020 | 0.954 ± 0.040 | 0.954 ± 0.040 | 1.000 ± 0.000 | 0.999 ± 0.001 | 0 |
| Random Forest | sem | `rf|n=300|prof=None|sem_escala` | 0.983 ± 0.012 | 0.946 ± 0.026 | 0.892 ± 0.054 | 0.892 ± 0.054 | 1.000 ± 0.000 | 0.997 ± 0.001 | 0 |
| Random Forest | com | `rf|n=300|prof=None|escala` | 0.983 ± 0.012 | 0.946 ± 0.026 | 0.892 ± 0.054 | 0.892 ± 0.054 | 1.000 ± 0.000 | 0.997 ± 0.001 | 0 |

## Comparação entre os três modelos

| modelo | melhor configuracao | AP | diferenca para o vencedor | empate tecnico (< 1 dp) | limiar sugerido | origem do limiar | limiar P>=0.95 (media ± dp) |
|---|---|---|---|---|---|---|---|
| k-NN | `knn_frac|k=3|w=distance|m=manhattan|sem_escala` | 1.000 ± 0.000 | 0.000 | vencedor | 0.6728 | precisao>=0.95 | 0.6728 ± 0 |
| SVM RBF | `svm_rbf|C=10|gamma=1|sem_escala` | 0.997 ± 0.007 | 0.003 | sim | -0.222 | precisao>=0.95 | -0.1786 ± 0.1657 |
| Random Forest | `rf|n=300|prof=None|sem_escala` | 0.983 ± 0.012 | 0.017 | não | 0.3967 | precisao>=0.95 | 0.3997 ± 0.06695 |

**Maior AP:** k-NN. Empate técnico = diferença de AP menor que o maior dos dois desvios-padrão.

Configurações empatadas com a maior AP de cada modelo (a escolhida é a primeira na ordem da grade): k-NN 2, SVM RBF 2, Random Forest 4.

**Regra de desempate sugerida:** se houver empate técnico, preferir o modelo mais simples e interpretável, na ordem k-NN (decisão explicável pelos vizinhos, que têm nome) → Random Forest (importância das características) → SVM RBF (fronteira no espaço do kernel). Por essa regra a sugestão seria **k-NN**. **A decisão final é da equipe.**

**Limiares:** o SVM usa `decision_function`, cuja escala muda a cada ajuste; o limiar transfere para o modelo treinado na semente inteira só de forma aproximada. A pontuação do k-NN por fração é discreta (poucos valores possíveis com k pequeno).

## Diagnóstico no oculto (não usado na escolha)

Melhor configuração de cada modelo treinada na semente inteira; "encontrado" = pontuação ≥ limiar sugerido do modelo. Famílias marcadas com * não têm morcego na semente.

| modelo | AP no oculto | morcegos encontrados | falsos positivos (mamíferos) |
|---|---|---|---|
| k-NN | 0.985 | 56/61 | 0 |
| SVM RBF | 1.000 | 57/61 | 0 |
| Random Forest | 0.994 | 55/61 | 0 |

### Recall por família

| familia | n | k-NN | SVM RBF | Random Forest |
|---|---|---|---|---|
| Hipposideridae* | 1 | 0 (0%) | 1 (100%) | 0 (0%) |
| Molossidae* | 2 | 2 (100%) | 2 (100%) | 1 (50%) |
| Mormoopidae* | 3 | 3 (100%) | 3 (100%) | 3 (100%) |
| Mystacinidae* | 1 | 0 (0%) | 1 (100%) | 1 (100%) |
| Phyllostomidae | 7 | 7 (100%) | 7 (100%) | 7 (100%) |
| Pteropodidae | 2 | 2 (100%) | 2 (100%) | 2 (100%) |
| Rhinolophidae* | 10 | 7 (70%) | 6 (60%) | 10 (100%) |
| Vespertilionidae | 35 | 35 (100%) | 35 (100%) | 31 (89%) |

### Recall por gênero

| familia | genero | n | k-NN | SVM RBF | Random Forest |
|---|---|---|---|---|---|
| Hipposideridae* | Hipposideros | 1 | 0 (0%) | 1 (100%) | 0 (0%) |
| Molossidae* | Tadarida | 2 | 2 (100%) | 2 (100%) | 1 (50%) |
| Mormoopidae* | Pteronotus | 3 | 3 (100%) | 3 (100%) | 3 (100%) |
| Mystacinidae* | Mystacina | 1 | 0 (0%) | 1 (100%) | 1 (100%) |
| Phyllostomidae | Artibeus | 2 | 2 (100%) | 2 (100%) | 2 (100%) |
| Phyllostomidae | Brachyphylla | 1 | 1 (100%) | 1 (100%) | 1 (100%) |
| Phyllostomidae | Lophostoma | 1 | 1 (100%) | 1 (100%) | 1 (100%) |
| Phyllostomidae | Micronycteris | 1 | 1 (100%) | 1 (100%) | 1 (100%) |
| Phyllostomidae | Sturnira | 1 | 1 (100%) | 1 (100%) | 1 (100%) |
| Phyllostomidae | Vampyrum | 1 | 1 (100%) | 1 (100%) | 1 (100%) |
| Pteropodidae | Cynopterus | 1 | 1 (100%) | 1 (100%) | 1 (100%) |
| Pteropodidae | Epomophorus | 1 | 1 (100%) | 1 (100%) | 1 (100%) |
| Rhinolophidae* | Rhinolophus | 10 | 7 (70%) | 6 (60%) | 10 (100%) |
| Vespertilionidae | Eptesicus | 1 | 1 (100%) | 1 (100%) | 1 (100%) |
| Vespertilionidae | Murina | 2 | 2 (100%) | 2 (100%) | 0 (0%) |
| Vespertilionidae | Myotis | 28 | 28 (100%) | 28 (100%) | 28 (100%) |
| Vespertilionidae | Pipistrellus | 2 | 2 (100%) | 2 (100%) | 2 (100%) |
| Vespertilionidae | Plecotus | 2 | 2 (100%) | 2 (100%) | 0 (0%) |

### Falsos positivos entre mamíferos não-morcegos do oculto

_(nenhum)_

### Vizinhos do k-NN

Configuração `knn_frac|k=3|w=distance|m=manhattan|sem_escala`; detalhes em `results/etapa2_oculto_vizinhos.csv` (3 vizinhos mais próximos na semente de cada morcego do oculto).

| familia | resultado | morcegos | media_vizinhos_morcego |
|---|---|---|---|
| Hipposideridae | perdido | 1 | 1.0 |
| Molossidae | encontrado | 2 | 3.0 |
| Mormoopidae | encontrado | 3 | 3.0 |
| Mystacinidae | perdido | 1 | 0.0 |
| Phyllostomidae | encontrado | 7 | 3.0 |
| Pteropodidae | encontrado | 2 | 3.0 |
| Rhinolophidae | encontrado | 7 | 2.57 |
| Rhinolophidae | perdido | 3 | 2.0 |
| Vespertilionidae | encontrado | 35 | 3.0 |

## Importância das características (melhor Random Forest)

Configuração `rf|n=300|prof=None|sem_escala`; 10 primeiras (todas em `results/etapa2_importancias_rf.csv`).

| posicao | caracteristica | importancia_semente | importancia_cv_media | importancia_cv_dp |
|---|---|---|---|---|
| 1 | c31 | 0.1796 | 0.1656 | 0.009 |
| 2 | c33 | 0.1424 | 0.1348 | 0.0171 |
| 3 | c48 | 0.1049 | 0.1077 | 0.0232 |
| 4 | c29 | 0.0671 | 0.0607 | 0.0189 |
| 5 | c46 | 0.0614 | 0.0624 | 0.0253 |
| 6 | c41 | 0.0512 | 0.0466 | 0.0147 |
| 7 | c02 | 0.0401 | 0.04 | 0.0113 |
| 8 | c20 | 0.0381 | 0.0386 | 0.0117 |
| 9 | c38 | 0.0344 | 0.0359 | 0.0128 |
| 10 | c27 | 0.0296 | 0.0274 | 0.0087 |

## Limites

- **Saturação:** várias configurações ficam com AP perto de 1 e as diferenças entre modelos são da ordem de 0,01, então a validação mal distingue os modelos. Explicação mais provável (não comprovada): os morcegos formam um grupo bem separado no espaço das 50 características e há poucos positivos. Dos 536 negativos da semente, os negativos prováveis (249, 46%) são fáceis de separar, mas os mamíferos não-morcegos (287, 54%) também ficam bem separados. O comportamento nas famílias sem representante (diagnóstico do oculto) diz mais sobre a prospecção.
- A Random Forest dá o mesmo resultado sem e com escala (árvores não dependem da escala das colunas), como esperado.
- Só 13 positivos de 9 gêneros na semente: cada dobra valida com 1 a 2 gêneros de morcego, então as métricas são ruidosas (ver desvios e boxplot).
- Os negativos prováveis da semente não foram verificados taxonomicamente (ruído de rótulo possível).
- O oculto concentra a maioria dos morcegos em *Myotis* (28) e *Rhinolophus* (10): acertar ou errar um desses gêneros muda o recall agregado de forma abrupta.
- Rhinolophidae, Hipposideridae, Molossidae, Mormoopidae e Mystacinidae não têm representante na semente.
- Precisão ≥ 0,95 com 13 positivos depende de pouquíssimos exemplos; o limiar é uma sugestão inicial para a Etapa 3, a ser revisto a cada rodada.
- O oculto não escolhe nada aqui; usá-lo para escolher invalidaria a medida de prospecção da Etapa 3.

## Figuras

- `results/figuras/etapa2_curvas_pr.png`
- `results/figuras/etapa2_boxplot_ap.png`
- `results/figuras/etapa2_recall_familia_oculto.png`

## Execução

Tempo total: 228.9 s.
