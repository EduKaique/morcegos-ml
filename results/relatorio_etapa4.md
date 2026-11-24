# Relatório da Etapa 4 — modelo final e testes de performance

Índices na base 0. O limiar e os hiperparâmetros são fixos (decididos na Etapa 2); nada aqui foi ajustado olhando o teste.

## Configuração

- Classificador (`src/config_modelo.py`, origem `results/relatorio_etapa2.md`): `knn_frac|n_neighbors=3|weights=distance|metric=manhattan|sem_escala`. Limiar fixo: **0.6728**.
- Modelo final treinado com `data/rounds/rodada_03/base.csv` (rodada manual 3, convergida): 624 instâncias, 74 morcegos e 550 negativos.

| origem | rotulo | n |
|---|---|---|
| gabarito | 0 | 10 |
| gabarito | 1 | 61 |
| humano | 0 | 4 |
| semente | 0 | 536 |
| semente | 1 | 13 |

- Teste congelado: 189 instâncias (5 morcegos, 184 negativos); SHA256 conferido (`68bc13a773dcaeee…`). Cada avaliação fica registrada em `results/etapa4_log_uso_teste.csv`.
- Base final e teste verificados sem interseção de índices nem de gêneros bloqueados.

## Teste congelado

### Matriz de confusão

| real | previsto_morcego | previsto_outro |
|---|---|---|
| morcego (1) | 5 | 0 |
| outro (0) | 1 | 183 |

### Métricas no limiar fixo (IC 95%)

Wilson para as proporções; bootstrap estratificado (positivos e negativos reamostrados separadamente, 2000 vezes, `random_state=42`) para F1 e AP, que não são proporções binomiais.

| métrica | valor [IC 95%] | k/n | método |
|---|---|---|---|
| sensibilidade | 1.000 [0.566; 1.000] | 5/5 | wilson |
| especificidade | 0.995 [0.970; 0.999] | 183/184 | wilson |
| precisao | 0.833 [0.436; 0.970] | 5/6 | wilson |
| acuracia | 0.995 [0.971; 0.999] | 188/189 | wilson |
| f1 | 0.909 [0.769; 1.000] | 10/11 | bootstrap_estratificado_2000 |
| precisao_media_ap | 0.833 [0.625; 1.000] | n=189 | bootstrap_estratificado_2000 |
| acuracia_linha_base_tudo_0 | 0.974 [0.940; 0.989] | 184/189 | wilson |

**Linha de base "tudo 0":** acurácia 0.974 [0.940; 0.989] sem reconhecer nenhum morcego. A acurácia do modelo (0.995 [0.971; 0.999]) mal se distingue dela: com 184 negativos em 189, a acurácia sozinha engana. Sensibilidade e precisão são as medidas que importam.

**AP e empates:** a AP (0.833) é calculada sobre pontuações com empates; negativos com a mesma pontuação de um morcego entram no mesmo degrau da curva e baixam a AP mesmo que todos os morcegos estejam acima do limiar.

### Negativos por origem

| origem | verificado | n | fp | especificidade |
|---|---|---|---|---|
| mamifero_por_genero | True | 133 | 1 | 0.992 [0.959; 0.999] |
| negativo_provavel | False | 51 | 0 | 1.000 [0.930; 1.000] |

`negativo_provavel` **não foi verificado taxonomicamente** (gênero fora da lista de mamíferos); `mamifero_por_genero` foi.

### Falsos positivos por ordem

| ordem | n_negativos | fp |
|---|---|---|
| Eulipotyphla | 3 | 1 |
| (sem ordem: negativo provável) | 51 | 0 |
| Primates | 32 | 0 |
| Artiodactyla | 28 | 0 |
| Carnivora | 21 | 0 |
| Rodentia | 16 | 0 |
| Cingulata | 12 | 0 |
| Didelphimorphia | 3 | 0 |
| Diprotodontia | 3 | 0 |
| Pilosa | 3 | 0 |
| Afrosoricida | 2 | 0 |
| Proboscidea | 2 | 0 |
| Dasyuromorphia | 1 | 0 |
| Hyracoidea | 1 | 0 |
| Lagomorpha | 1 | 0 |
| Monotremata | 1 | 0 |
| Paucituberculata | 1 | 0 |
| Peramelemorphia | 1 | 0 |
| Perissodactyla | 1 | 0 |
| Sirenia | 1 | 0 |

## Conjunto supervisionado final (base final ∪ teste)

813 instâncias, 79 morcegos de 32 gêneros. Origem dos rótulos:

| origem_rotulo | rotulo | n |
|---|---|---|
| chiroptera_por_genero | 1 | 18 |
| gabarito_taxonomia | 0 | 10 |
| gabarito_taxonomia | 1 | 61 |
| humano | 0 | 4 |
| mamifero_por_genero | 0 | 420 |
| negativo_provavel | 0 | 300 |

**Aviso:** os 300 `negativo_provavel` deste conjunto não são verificados.

### Validação agrupada por gênero (5 dobras × 10 repetições, sementes 42..51)

Pontuações fora da amostra por repetição; métricas no limiar fixo; AP sem limiar.

| métrica | media | dp |
|---|---|---|
| sensibilidade | 0.951 | 0.018 |
| especificidade | 1.0 | 0.001 |
| precisao | 0.996 | 0.009 |
| acuracia | 0.995 | 0.002 |
| f1 | 0.973 | 0.009 |
| ap | 0.993 | 0.009 |

Matriz de confusão agregada da repetição com semente 42:

| real | previsto_morcego | previsto_outro |
|---|---|---|
| morcego (1) | 75 | 4 |
| outro (0) | 0 | 734 |

### Leave-one-genus-out dos gêneros de morcego

Para cada gênero de morcego, o modelo é treinado com todo o resto do conjunto e pontua as espécies do gênero retirado: mede o reconhecimento de gêneros nunca vistos.

- Gêneros reconhecidos por inteiro: 29 de 32; espécies reconhecidas: 75 de 79.

Por família:

| familia | generos | especies | reconhecidas | generos_inteiros |
|---|---|---|---|---|
| Emballonuridae | 1 | 1 | 1 | 1 |
| Hipposideridae | 1 | 1 | 0 | 0 |
| Molossidae | 1 | 2 | 2 | 1 |
| Mormoopidae | 1 | 3 | 3 | 1 |
| Mystacinidae | 1 | 1 | 0 | 0 |
| Phyllostomidae | 11 | 12 | 12 | 11 |
| Pteropodidae | 4 | 7 | 7 | 4 |
| Rhinolophidae | 1 | 10 | 8 | 0 |
| Vespertilionidae | 11 | 42 | 42 | 11 |

Gêneros com espécies não reconhecidas:

| genero | familia | n | reconhecidos | pontuacao_media | nao_reconhecidos |
|---|---|---|---|---|---|
| Hipposideros | Hipposideridae | 1 | 0 | 0.335 | Hipposideros_armiger |
| Mystacina | Mystacinidae | 1 | 0 | 0.357 | Mystacina_tuberculata |
| Rhinolophus | Rhinolophidae | 10 | 8 | 0.901 | Rhinolophus_formosae;Rhinolophus_luctus |

## Análise de erros

Arquivo completo: `results/etapa4_erros.csv` (com os 3 vizinhos mais próximos: na base final para o teste; no conjunto sem o próprio gênero para a validação agrupada).

### Falsos positivos no teste

| tipo_erro | indice | nome | ordem | familia | pontuacao | viz1_nome | viz1_classe | viz2_nome | viz2_classe | viz3_nome | viz3_classe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FP | 2916 | Galemys_pyrenaicus | Eulipotyphla | Talpidae | 1.0 | Cynopterus_brachyotis | 1 | Saccopteryx_leptura | 1 | Pteropus_scapulatus | 1 |

### Falsos negativos no teste

_(nenhum)_

### Quase-erros: 10 negativos de maior pontuação

| tipo_erro | indice | nome | ordem | familia | pontuacao | viz1_nome | viz1_classe | viz2_nome | viz2_classe | viz3_nome | viz3_classe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| quase_erro_neg | 2916 | Galemys_pyrenaicus | Eulipotyphla | Talpidae | 1.0 | Cynopterus_brachyotis | 1 | Saccopteryx_leptura | 1 | Pteropus_scapulatus | 1 |
| quase_erro_neg | 7012 | Scapanulus_oweni | Eulipotyphla | Talpidae | 0.666 | Cynopterus_brachyotis | 1 | Mogera_robusta | 0 | Hipposideros_armiger | 1 |
| quase_erro_neg | 7977 | Trichechus_manatus | Sirenia | Trichechidae | 0.338 | Micronycteris_megalotis | 1 | Priodontes_maximus | 0 | Carlito_syrichta | 0 |
| quase_erro_neg | 134 | Acrocephalus_scirpaceus |  |  | 0.0 | Zosterops_lateralis | 0 | Pyrgilauda_davidiana | 0 | Regulus_regulus | 0 |
| quase_erro_neg | 347 | Ammodytes_americanus |  |  | 0.0 | Labracoglossa_argenteiventris | 0 | Ditrema_temminckii | 0 | Macquaria_ambigua | 0 |
| quase_erro_neg | 551 | Antidorcas_marsupialis | Artiodactyla | Bovidae | 0.0 | Cervus_nippon_kopschi | 0 | Cervus_nippon_sichuanicus | 0 | Cervus_nippon_hortulorum | 0 |
| quase_erro_neg | 618 | Apostichopus_japonicus |  |  | 0.0 | Sooglossus_thomasseti | 0 | Atrina_pectinata | 0 | Mengenilla_moldrzyki | 0 |
| quase_erro_neg | 657 | Arconaia_lanceolata |  |  | 0.0 | Margaritifera_dahurica | 0 | Physella_acuta | 0 | Nectonemertes_cf_mirabilis_HC_2011 | 0 |
| quase_erro_neg | 665 | Arctonyx_collaris | Carnivora | Mustelidae | 0.0 | Meles_anakuma | 0 | Meles_meles | 0 | Melogale_moschata | 0 |
| quase_erro_neg | 714 | Asbolus_verrucosus |  |  | 0.0 | Eucryptorrhynchus_chinensis | 0 | Epicauta_tibialis | 0 | Sitophilus_oryzae | 0 |

### Quase-erros: positivos de menor pontuação (até 10)

| tipo_erro | indice | nome | ordem | familia | pontuacao | viz1_nome | viz1_classe | viz2_nome | viz2_classe | viz3_nome | viz3_classe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| quase_erro_pos | 531 | Anoura_caudifer | Chiroptera | Phyllostomidae | 1.0 | Vampyrum_spectrum | 1 | Lophostoma_silvicolum | 1 | Brachyphylla_cavernarum | 1 |
| quase_erro_pos | 3668 | Hypsugo_alaschanicus | Chiroptera | Vespertilionidae | 1.0 | Pipistrellus_abramus | 1 | Vespertilio_murinus | 1 | Vespertilio_sinensis | 1 |
| quase_erro_pos | 3941 | Lasiurus_borealis | Chiroptera | Vespertilionidae | 1.0 | Myotis_martiniquensis | 1 | Myotis_formosus | 1 | Myotis_auriculus | 1 |
| quase_erro_pos | 5198 | Nyctalus_noctula | Chiroptera | Vespertilionidae | 1.0 | Vespertilio_sinensis | 1 | Pipistrellus_coromandra | 1 | Vespertilio_murinus | 1 |
| quase_erro_pos | 6847 | Rousettus_aegyptiacus | Chiroptera | Pteropodidae | 1.0 | Pteropus_vampyrus | 1 | Pteropus_dasymallus | 1 | Cynopterus_brachyotis | 1 |

### Erros da validação agrupada (em ≥ 1 das 10 repetições)

| tipo_erro | indice | nome | ordem | familia | pontuacao | n_rep_erro | viz1_nome | viz1_classe |
|---|---|---|---|---|---|---|---|---|
| FN | 3478 | Hipposideros_armiger | Chiroptera | Hipposideridae | 0.397 | 10 | Scapanulus_oweni | 0 |
| FN | 4912 | Mystacina_tuberculata | Chiroptera | Mystacinidae | 0.45 | 8 | Cynopterus_brachyotis | 1 |
| FN | 6743 | Rhinolophus_formosae | Chiroptera | Rhinolophidae | 0.729 | 8 | Urotrichus_talpoides | 0 |
| FN | 6744 | Rhinolophus_luctus | Chiroptera | Rhinolophidae | 0.765 | 7 | Hipposideros_armiger | 1 |
| FN | 2290 | Desmodus_rotundus | Chiroptera | Phyllostomidae | 0.87 | 2 | Carollia_perspicillata | 1 |
| FN | 6847 | Rousettus_aegyptiacus | Chiroptera | Pteropodidae | 0.81 | 2 | Pteropus_vampyrus | 1 |
| FN | 2159 | Cynopterus_brachyotis | Chiroptera | Pteropodidae | 0.933 | 1 | Rousettus_aegyptiacus | 1 |
| FN | 6512 | Pteropus_alecto | Chiroptera | Pteropodidae | 0.966 | 1 | Rousettus_aegyptiacus | 1 |
| FP | 7012 | Scapanulus_oweni | Eulipotyphla | Talpidae | 0.635 | 2 | Cynopterus_brachyotis | 1 |
| FP | 2916 | Galemys_pyrenaicus | Eulipotyphla | Talpidae | 0.696 | 1 | Scapanulus_oweni | 0 |

Nos quase-erros, negativos com a mesma pontuação estão em ordem de índice; com k=3, 181 dos 184 negativos do teste têm pontuação exatamente 0.

## Limites

- **O teste tem só 5 morcegos.** Cada um vale 20% da sensibilidade; mesmo 5 de 5 deixa o limite inferior do IC de Wilson em 56,6%. A sensibilidade do teste é pouco informativa sozinha.
- **Os 51 negativos prováveis do teste não foram verificados**: se algum for morcego fora da lista, um "falso positivo" pode ser acerto (e vice-versa).
- **O k-NN com k=3 dá pontuação discreta** (poucos valores possíveis): a AP e as curvas têm poucos degraus e o bootstrap da AP é grosseiro.
- **A avaliação no conjunto supervisionado final não mede o enriquecimento sem rótulos**, só o classificador com os rótulos verificados. Os negativos prováveis desse conjunto também não são verificados.
- O teste congelado é avaliado só com `--avaliar-teste`; repetir a avaliação enquanto se altera o modelo invalida o teste. Histórico em `results/etapa4_log_uso_teste.csv`.

## Figuras

- `results/figuras/etapa4_matriz_confusao_teste.png`
- `results/figuras/etapa4_pontuacoes_teste.png`
- `results/figuras/etapa4_ic_wilson_teste.png`
- `results/figuras/etapa4_logo_generos.png`
