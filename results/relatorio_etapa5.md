# Relatório da Etapa 5 — entrega das 8426 classes

Este relatório não altera nenhum rótulo e não avalia o teste congelado: nenhuma métrica é calculada sobre o teste e nada é gravado em `results/etapa4_log_uso_teste.csv`. O teste só é lido (com o SHA256 conferido) para saber quais índices pertencem a ele.

## Configuração

- Arquivos entregues: `results/classes_finais.csv` (8426 linhas: `indice_base0`, `indice_base1`, `nome`, `classe`, `fonte_rotulo`, `pontuacao`) e `results/indices_positivos.txt` (79 linhas), gerados por `scripts/gerar_classes_finais.py`.
- Modelo final: `knn_frac|n_neighbors=3|weights=distance|metric=manhattan|sem_escala` (`src/config_modelo.py`), treinado na base da rodada manual 3 (convergida, `data/rounds/rodada_03/base.csv`: 624 instâncias, 74 morcegos). Limiar fixo: **0.6728** (Etapa 2).
- Prioridade do rótulo: gabarito taxonômico → teste verificado → base final → modelo final (pontuação ≥ limiar). O modelo só decide onde não há rótulo verificado.
- **Base dos índices:** o enunciado pede os *números de ordem* dos morcegos, contados a partir de 1 (a primeira linha de `Names.txt` é o organismo 1). Por isso `indices_positivos.txt` usa a **base 1** (`indice_base1`). Internamente o projeto usa base 0 (`indice` = `indice_base0`); `indice_base1 = indice_base0 + 1`. As tabelas abaixo mostram `indice_base0`.

## Verificações de consistência

Todas passaram (o comando para com erro se alguma falhar):

- 8426 linhas, `indice_base0` único e cobrindo 0..8425; `indice_base1 = indice_base0 + 1`;
- `classe` só 0 ou 1;
- `indices_positivos.txt` tem exatamente os 79 positivos de `classes_finais.csv` na base 1;
- os 79 positivos vêm do gabarito; a soma por fonte é 8426;
- nenhum índice do teste ou de gênero bloqueado (252 índices) aparece nos candidatos das rodadas manuais 1..3 (75 candidatos) nem na base final;
- as contagens da tabela abaixo são relidas deste relatório e comparadas com o CSV.

## Fonte do rótulo x classe

| fonte | classe 0 | classe 1 | total |
|---|---|---|---|
| gabarito | 921 | 79 | 1000 |
| teste_verificado | 51 | 0 | 51 |
| base_final | 253 | 0 | 253 |
| modelo_final | 7122 | 0 | 7122 |
| total | 8347 | 79 | 8426 |

**79 morcegos**, todos com rótulo do gabarito taxonômico (MDD v2.5, ordem Chiroptera). Positivos decididos só pelo modelo: **0**.

## Morcegos por família e gênero

Arquivo: `results/etapa5_morcegos.csv` (uma linha por morcego; gênero e família do gabarito). `origem_entrada` = quando o morcego entrou na base de treino: `semente` (rodada 0), `rodada_N` (coluna `rodada` de `data/rounds/rodada_03/base.csv`) ou `teste_bloqueado` (teste congelado; nunca entrou no treino).

| origem_entrada | morcegos |
|---|---|
| semente | 13 |
| rodada_1 | 60 |
| rodada_2 | 1 |
| rodada_3 | 0 |
| teste_bloqueado | 5 |

### Por família

| familia | generos | morcegos | origem |
|---|---|---|---|
| Vespertilionidae | 11 | 42 | semente 4; rodada 1 35; teste bloqueado 3 |
| Phyllostomidae | 11 | 12 | semente 4; rodada 1 7; teste bloqueado 1 |
| Rhinolophidae | 1 | 10 | rodada 1 10 |
| Pteropodidae | 4 | 7 | semente 4; rodada 1 2; teste bloqueado 1 |
| Mormoopidae | 1 | 3 | rodada 1 3 |
| Molossidae | 1 | 2 | rodada 1 2 |
| Emballonuridae | 1 | 1 | semente 1 |
| Hipposideridae | 1 | 1 | rodada 1 1 |
| Mystacinidae | 1 | 1 | rodada 2 1 |

### Por gênero

| familia | genero | n | origem | especies |
|---|---|---|---|---|
| Emballonuridae | Saccopteryx | 1 | semente 1 | Saccopteryx_leptura |
| Hipposideridae | Hipposideros | 1 | rodada 1 1 | Hipposideros_armiger |
| Molossidae | Tadarida | 2 | rodada 1 2 | Tadarida_latouchei, Tadarida_teniotis |
| Mormoopidae | Pteronotus | 3 | rodada 1 3 | Pteronotus_parnellii, Pteronotus_personatus, Pteronotus_rubiginosus |
| Mystacinidae | Mystacina | 1 | rodada 2 1 | Mystacina_tuberculata |
| Phyllostomidae | Anoura | 1 | teste bloqueado 1 | Anoura_caudifer |
| Phyllostomidae | Artibeus | 2 | rodada 1 2 | Artibeus_jamaicensis, Artibeus_lituratus |
| Phyllostomidae | Brachyphylla | 1 | rodada 1 1 | Brachyphylla_cavernarum |
| Phyllostomidae | Carollia | 1 | semente 1 | Carollia_perspicillata |
| Phyllostomidae | Desmodus | 1 | semente 1 | Desmodus_rotundus |
| Phyllostomidae | Lophostoma | 1 | rodada 1 1 | Lophostoma_silvicolum |
| Phyllostomidae | Micronycteris | 1 | rodada 1 1 | Micronycteris_megalotis |
| Phyllostomidae | Rhinophylla | 1 | semente 1 | Rhinophylla_pumilio |
| Phyllostomidae | Sturnira | 1 | rodada 1 1 | Sturnira_tildae |
| Phyllostomidae | Tonatia | 1 | semente 1 | Tonatia_saurophila |
| Phyllostomidae | Vampyrum | 1 | rodada 1 1 | Vampyrum_spectrum |
| Pteropodidae | Cynopterus | 1 | rodada 1 1 | Cynopterus_brachyotis |
| Pteropodidae | Epomophorus | 1 | rodada 1 1 | Epomophorus_gambianus |
| Pteropodidae | Pteropus | 4 | semente 4 | Pteropus_alecto, Pteropus_dasymallus, Pteropus_scapulatus, Pteropus_vampyrus |
| Pteropodidae | Rousettus | 1 | teste bloqueado 1 | Rousettus_aegyptiacus |
| Rhinolophidae | Rhinolophus | 10 | rodada 1 10 | Rhinolophus_ferrumequinum_korai, Rhinolophus_ferrumequinum_quelpartis, Rhinolophus_formosae, Rhinolophus_luctus, Rhinolophus_macrotis, Rhinolophus_monoceros, Rhinolophus_pumilus, Rhinolophus_rex, Rhinolophus_thomasi, Rhinolophus_yunnanensis |
| Vespertilionidae | Chalinolobus | 1 | semente 1 | Chalinolobus_tuberculatus |
| Vespertilionidae | Corynorhinus | 1 | semente 1 | Corynorhinus_rafinesquii |
| Vespertilionidae | Eptesicus | 1 | rodada 1 1 | Eptesicus_serotinus |
| Vespertilionidae | Hypsugo | 1 | teste bloqueado 1 | Hypsugo_alaschanicus |
| Vespertilionidae | Lasiurus | 1 | teste bloqueado 1 | Lasiurus_borealis |
| Vespertilionidae | Murina | 2 | rodada 1 2 | Murina_leucogaster, Murina_ussuriensis |
| Vespertilionidae | Myotis | 28 | rodada 1 28 | Myotis_albescens, Myotis_atacamensis, Myotis_auriculus, Myotis_bechsteinii, Myotis_bombinus, Myotis_brandtii, Myotis_davidii, Myotis_dominicensis, Myotis_evotis, Myotis_formosus, Myotis_horsfieldii, Myotis_ikonnikovi, Myotis_keaysi, Myotis_leibii, Myotis_lucifugus, Myotis_macrodactylus, Myotis_martiniquensis, Myotis_melanorhinus, Myotis_muricola, Myotis_myotis, Myotis_nigricans, Myotis_oxyotus, Myotis_petax, Myotis_riparius, Myotis_ruber, Myotis_thysanodes, Myotis_volans, Myotis_yumanensis |
| Vespertilionidae | Nyctalus | 1 | teste bloqueado 1 | Nyctalus_noctula |
| Vespertilionidae | Pipistrellus | 2 | rodada 1 2 | Pipistrellus_abramus, Pipistrellus_coromandra |
| Vespertilionidae | Plecotus | 2 | rodada 1 2 | Plecotus_auritus, Plecotus_macrobullaris |
| Vespertilionidae | Vespertilio | 2 | semente 2 | Vespertilio_murinus, Vespertilio_sinensis |

## O que o modelo decidiu

O modelo final decidiu sozinho 7122 instâncias (sem rótulo em nenhuma fonte verificada). **Nenhuma passou do limiar: zero positivos vieram só do modelo.**

| pontuação | instâncias |
|---|---|
| exatamente 0 | 7122 |
| entre 0 e o limiar (0.6728) | 0 |
| ≥ limiar | 0 |

Com k=3 e votos por distância, a pontuação só é maior que 0 quando ao menos um dos 3 vizinhos mais próximos na base final é morcego. Como todas as pontuações são iguais, a ordem das "20 de maior pontuação" foi desempatada pela proximidade `d_neg / (d_pos + d_neg)` (manhattan, sem escala, a variante `knn_razao` da Etapa 2): d_pos e d_neg são as distâncias ao morcego e ao não-morcego mais próximos da base final; 1 = em cima de um morcego, 0,5 = a igual distância. **Esse desempate só ordena a tabela; não muda nenhuma classe.**

### As 20 sem rótulo de maior pontuação (desempate pela proximidade)

| indice_base0 | nome | genero | pontuacao | proximidade | gênero bloqueado | vizinhos na base final (classe) |
|---|---|---|---|---|---|---|
| 8225 | Utricularia_reniformis | Utricularia | 0.0 | 0.483 | não | Cucurbita_pepo (0); Perionyx_excavatus (0); Marphysa_sanguinea (0) |
| 6773 | Rhizophagus_intraradices | Rhizophagus | 0.0 | 0.4794 | não | Atrina_pectinata (0); Uroctonus_mordax (0); Dorypteryx_domestica (0) |
| 7995 | Trichoderma_harzianum | Trichoderma | 0.0 | 0.4787 | não | Mengenilla_moldrzyki (0); Nectonemertes_cf_mirabilis_HC_2011 (0); Aleuroglyphus_ovatus (0) |
| 2588 | Epigonichthys_cultellus | Epigonichthys | 0.0 | 0.4767 | não | Gekko_swinhonis (0); Argas_brumpti (0); Neosynchiropus_moyeri (0) |
| 1259 | Candida_alimentaria | Candida | 0.0 | 0.4762 | não | Mengenilla_moldrzyki (0); Ogataea_philodendri (0); Candida_theae (0) |
| 7963 | Treubia_lacunosa | Treubia | 0.0 | 0.4757 | não | Helicosporidium_sp_ex_Simulium_jonesi (0); Coccomyxa_subellipsoidea_C_169 (0); Chlorella_sorokiniana (0) |
| 7814 | Thecamonas_trahens | Thecamonas | 0.0 | 0.4727 | não | Megacopta_cribraria (0); Argas_brumpti (0); Botrylloides_giganteum (0) |
| 2322 | Dictyota_dichotoma | Dictyota | 0.0 | 0.4726 | não | Ornithodoros_porcinus (0); Paracercomonas_marina (0); Potamyia_flava (0) |
| 3721 | Indotyphlops_braminus | Indotyphlops | 0.0 | 0.4699 | não | Erinaceus_europaeus (0); Marphysa_sanguinea (0); Heterocephalus_glaber (0) |
| 2289 | Desmarestia_viridis | Desmarestia | 0.0 | 0.4694 | não | Sargassum_vachellianum (0); Saccharina_angustata (0); Choloepus_hoffmanni (0) |
| 4237 | Loxechinus_albus | Loxechinus | 0.0 | 0.4682 | não | Atrina_pectinata (0); Sooglossus_thomasseti (0); Leiolepis_boehmei (0) |
| 6056 | Plasmodium_chabaudi_chabaudi | Plasmodium | 0.0 | 0.4678 | não | Atrina_pectinata (0); Bradypus_pygmaeus (0); Acanthocheilonema_viteae (0) |
| 2628 | Eptatretus_burgeri | Eptatretus | 0.0 | 0.4672 | não | Choloepus_didactylus (0); Megacopta_cribraria (0); Bradypus_pygmaeus (0) |
| 5982 | Phycomyces_blakesleeanus | Phycomyces | 0.0 | 0.467 | não | Naumovozyma_castellii (0); Ogataea_philodendri (0); Rhizomys_pruinosus (0) |
| 6120 | Pleurozia_purpurea | Pleurozia | 0.0 | 0.4655 | não | Coccomyxa_subellipsoidea_C_169 (0); Ophioglossum_californicum (0); Helicosporidium_sp_ex_Simulium_jonesi (0) |
| 739 | Aspergillus_ustus | Aspergillus | 0.0 | 0.465 | não | Madurella_mycetomatis (0); Plakinastrella_cf_onkodes_DVL_2011 (0); Physella_acuta (0) |
| 6848 | Roya_obtusa | Roya | 0.0 | 0.4644 | não | Helicosporidium_sp_ex_Simulium_jonesi (0); Coccomyxa_subellipsoidea_C_169 (0); Chlorella_sorokiniana (0) |
| 6774 | Rhizophagus_irregularis | Rhizophagus | 0.0 | 0.4631 | não | Uroctonus_mordax (0); Mengenilla_moldrzyki (0); Atrina_pectinata (0) |
| 423 | Ancoracysta_twista | Ancoracysta | 0.0 | 0.4631 | não | Sitophilus_oryzae (0); Eucryptorrhynchus_chinensis (0); Batocera_lineolata (0) |
| 6067 | Plasmodium_yoelii | Plasmodium | 0.0 | 0.463 | não | Bradypus_pygmaeus (0); Acanthocheilonema_viteae (0); Atrina_pectinata (0) |

Proximidade máxima no pool: 0.4830; instâncias com proximidade > 0,5: 0. Valores perto de 0,5 aparecem em organismos isolados, longe de toda a base (morcegos e não-morcegos quase à mesma distância), e não indicam semelhança com morcegos. Entre as 20, 0 têm algum morcego entre os 3 vizinhos. Gênero bloqueado = gênero reservado ao teste.

### Sem rótulo em gêneros bloqueados

Os gêneros do teste somam 252 instâncias: 145 com rótulo do gabarito, 51 negativos prováveis do teste e **56 sem rótulo**, decididas pelo modelo (todas fora do treino e do enriquecimento):

| indice_base0 | nome | genero | pontuacao | proximidade |
|---|---|---|---|---|
| 348 | Ammodytes_dubius | Ammodytes | 0.0 | 0.2901 |
| 349 | Ammodytes_hexapterus | Ammodytes | 0.0 | 0.2885 |
| 350 | Ammodytes_personatus | Ammodytes | 0.0 | 0.2996 |
| 619 | Apostichopus_parvimensis | Apostichopus | 0.0 | 0.4328 |
| 781 | Atractosteus_spatula | Atractosteus | 0.0 | 0.3766 |
| 783 | Atractosteus_tropicus | Atractosteus | 0.0 | 0.3794 |
| 1074 | Brassica_carinata | Brassica | 0.0 | 0.3923 |
| 1076 | Brassica_napus | Brassica | 0.0 | 0.4132 |
| 1077 | Brassica_nigra | Brassica | 0.0 | 0.4064 |
| 1078 | Brassica_oleracea | Brassica | 0.0 | 0.3987 |
| 1079 | Brassica_rapa_subsp_oleifera | Brassica | 0.0 | 0.4089 |
| 1093 | Brycon_orbignyanus | Brycon | 0.0 | 0.4115 |
| 1806 | Coilia_brachygnathus | Coilia | 0.0 | 0.3087 |
| 1807 | Coilia_grayii | Coilia | 0.0 | 0.3026 |
| 1808 | Coilia_lindmani | Coilia | 0.0 | 0.3105 |
| 1810 | Coilia_nasus | Coilia | 0.0 | 0.3074 |
| 1811 | Coilia_reynaldi | Coilia | 0.0 | 0.3019 |
| 1817 | Colletotrichum_acutatum | Colletotrichum | 0.0 | 0.3484 |
| 1819 | Colletotrichum_graminicola_M1001 | Colletotrichum | 0.0 | 0.3285 |
| 1820 | Colletotrichum_lindemuthianum | Colletotrichum | 0.0 | 0.3873 |
| 1821 | Colletotrichum_lupini | Colletotrichum | 0.0 | 0.348 |
| 1822 | Colletotrichum_salicis | Colletotrichum | 0.0 | 0.3457 |
| 1823 | Colletotrichum_tamarilloi | Colletotrichum | 0.0 | 0.3465 |
| 1909 | Cordyceps_brongniartii | Cordyceps | 0.0 | 0.2307 |
| 2809 | Favites_abdita | Favites | 0.0 | 0.2639 |
| 2838 | Floraphis_meitanensis | Floraphis | 0.0 | 0.2328 |
| 2877 | Furculitermes_cubitalis | Furculitermes | 0.0 | 0.2037 |
| 2879 | Furculitermes_soyeri | Furculitermes | 0.0 | 0.2104 |
| 2880 | Furculitermes_winifredae | Furculitermes | 0.0 | 0.2262 |
| 2897 | Gadopsis_marmoratus | Gadopsis | 0.0 | 0.3299 |
| 3220 | Gyrinophilus_palleucus | Gyrinophilus | 0.0 | 0.3289 |
| 3255 | Haliotis_discus_hannai_x_Haliotis_iris | Haliotis | 0.0 | 0.426 |
| 3259 | Haliotis_rufescens | Haliotis | 0.0 | 0.4382 |
| 3260 | Haliotis_tuberculata_tuberculata | Haliotis | 0.0 | 0.4054 |
| 3652 | Hypsibarbus_salweenensis | Hypsibarbus | 0.0 | 0.1691 |
| 4517 | Megalurus_pryeri | Megalurus | 0.0 | 0.2908 |
| 4583 | Mesobuthus_gibbosus | Mesobuthus | 0.0 | 0.4113 |
| 4963 | Nanorana_parkeri | Nanorana | 0.0 | 0.4011 |
| 4965 | Nanorana_taihangnica | Nanorana | 0.0 | 0.3792 |
| 5268 | Oesophagostomum_asperum | Oesophagostomum | 0.0 | 0.2039 |
| 5270 | Oesophagostomum_dentatum | Oesophagostomum | 0.0 | 0.1846 |
| 5271 | Oesophagostomum_quadrispinulatum | Oesophagostomum | 0.0 | 0.1808 |
| 5953 | Pholis_fangi | Pholis | 0.0 | 0.2248 |
| 5954 | Pholis_nebulosa | Pholis | 0.0 | 0.2209 |
| 7041 | Schistosoma_haematobium | Schistosoma | 0.0 | 0.3826 |
| 7043 | Schistosoma_mansoni | Schistosoma | 0.0 | 0.371 |
| 7044 | Schistosoma_mekongi | Schistosoma | 0.0 | 0.3822 |
| 7045 | Schistosoma_spindale | Schistosoma | 0.0 | 0.4045 |
| 7344 | Solemya_elarraichensis | Solemya | 0.0 | 0.3959 |
| 7346 | Solemya_velesiana | Solemya | 0.0 | 0.3963 |
| 7347 | Solemya_velum | Solemya | 0.0 | 0.3801 |
| 7766 | Tetrahymena_malaccensis | Tetrahymena | 0.0 | 0.4355 |
| 7768 | Tetrahymena_pigmentosa | Tetrahymena | 0.0 | 0.4269 |
| 7770 | Tetrahymena_thermophila | Tetrahymena | 0.0 | 0.4328 |
| 8268 | Vespa_orientalis | Vespa | 0.0 | 0.3741 |
| 8269 | Vespa_velutina | Vespa | 0.0 | 0.3795 |

### Sem rótulo com gênero suspeito

4 nomes cujo primeiro termo está em minúscula (não é gênero); nunca receberam rótulo:

| indice_base0 | nome | pontuacao | proximidade |
|---|---|---|---|
| 8422 | diploid_Megalobrama_amblycephala_x_Xenocypris_davidi | 0.0 | 0.1073 |
| 8423 | diploid_Xenocypris_davidi_x_Megalobrama_amblycephala | 0.0 | 0.1186 |
| 8424 | natural_gynogenetic_Carassius_auratus_red_var | 0.0 | 0.1839 |
| 8425 | triploid_Megalobrama_amblycephala_x_Xenocypris_davidi | 0.0 | 0.1073 |

## Rótulo verificado contra o modelo

1 instância(s) em que o rótulo verificado discorda do modelo no limiar:

| indice_base0 | nome | classe | fonte_rotulo | pontuacao | ordem | familia | vizinhos na base final (classe) |
|---|---|---|---|---|---|---|---|
| 2916 | Galemys_pyrenaicus | 0 | gabarito | 1.0 | Eulipotyphla | Talpidae | Cynopterus_brachyotis (1); Saccopteryx_leptura (1); Pteropus_scapulatus (1) |

O caso de *Galemys pyrenaicus* (desmã-dos-pirenéus, Eulipotyphla: Talpidae) mostra a prioridade funcionando: o modelo dá pontuação 1,0 porque os 3 vizinhos mais próximos na base final são morcegos, mas o gabarito taxonômico diz que não é morcego, e a classe final segue o gabarito (0). É o mesmo falso positivo do teste na Etapa 4.

## Segunda opinião (complementar)

**Isso é só um sinal adicional de ausência de candidatos escondidos; nenhum rótulo é alterado.** A melhor configuração de cada modelo na Etapa 2 (`results/etapa2_comparacao.csv`, maior AP, mesma regra de desempate da Etapa 2) foi treinada na base final e pontuou as 7122 instâncias sem rótulo. O limiar é o sugerido na Etapa 2 (mediana do menor limiar com precisão ≥ 0,95), que só transfere de forma aproximada para um modelo treinado em outra base (sobretudo o SVM, cuja `decision_function` muda de escala a cada ajuste).

| modelo | configuração | limiar sugerido | sem rótulo ≥ limiar | pontuação máxima | interseção com o top 20 do k-NN |
|---|---|---|---|---|---|
| SVM RBF | `svm_rbf\|C=10\|gamma=1\|sem_escala` | -0.222 | 0 | -0.6973 | 2 |
| Random Forest | `rf\|n=300\|prof=None\|sem_escala` | 0.3967 | 0 | 0.19 | 1 |

### SVM RBF: as 20 sem rótulo de maior pontuação

| indice_base0 | nome | genero | pontuacao | gênero bloqueado | no top k-NN |
|---|---|---|---|---|---|
| 1801 | Coelodonta_antiquitatis | Coelodonta | -0.6973 | não | não |
| 5763 | Pecari_tajacu | Pecari | -0.8069 | não | não |
| 3436 | Hexaprotodon_liberiensis | Hexaprotodon | -0.9065 | não | não |
| 5556 | Pampus_argenteus | Pampus | -0.9787 | não | não |
| 5559 | Pampus_echinogaster | Pampus | -0.9787 | não | não |
| 5562 | Pampus_sp_LY_2009 | Pampus | -0.9787 | não | não |
| 8225 | Utricularia_reniformis | Utricularia | -0.9835 | não | sim |
| 949 | Beta_macrocarpa | Beta | -0.9835 | não | não |
| 950 | Beta_vulgaris_subsp_maritima | Beta | -0.9835 | não | não |
| 951 | Beta_vulgaris_subsp_vulgaris | Beta | -0.9835 | não | não |
| 4429 | Marchantia_paleacea | Marchantia | -0.9835 | não | não |
| 1706 | Chrysoporthe_austroafricana | Chrysoporthe | -0.9835 | não | não |
| 1329 | Capsicum_annuum | Capsicum | -0.9835 | não | não |
| 1227 | Calypogeia_neogaea | Calypogeia | -0.9835 | não | não |
| 1076 | Brassica_napus | Brassica | -0.9835 | sim | não |
| 1228 | Calypogeia_suecica | Calypogeia | -0.9835 | não | não |
| 1226 | Calypogeia_integristipula | Calypogeia | -0.9835 | não | não |
| 8077 | Tritomaria_quinquedentata | Tritomaria | -0.9835 | não | não |
| 1225 | Calypogeia_arguta | Calypogeia | -0.9835 | não | não |
| 6120 | Pleurozia_purpurea | Pleurozia | -0.9835 | não | sim |

### Random Forest: as 20 sem rótulo de maior pontuação

| indice_base0 | nome | genero | pontuacao | gênero bloqueado | no top k-NN |
|---|---|---|---|---|---|
| 4318 | Lythrurus_lirus | Lythrurus | 0.19 | não | não |
| 3231 | Haemaphysalis_concinna | Haemaphysalis | 0.1867 | não | não |
| 7963 | Treubia_lacunosa | Treubia | 0.1733 | não | sim |
| 831 | Babina_subaspera | Babina | 0.1667 | não | não |
| 1648 | Chlorokybus_atmophyticus | Chlorokybus | 0.1667 | não | não |
| 1706 | Chrysoporthe_austroafricana | Chrysoporthe | 0.1667 | não | não |
| 830 | Babina_holsti | Babina | 0.1633 | não | não |
| 4429 | Marchantia_paleacea | Marchantia | 0.1633 | não | não |
| 1707 | Chrysoporthe_cubensis | Chrysoporthe | 0.16 | não | não |
| 2280 | Dermacentor_nitens | Dermacentor | 0.16 | não | não |
| 57 | Acerentomon_microrhinus | Acerentomon | 0.15 | não | não |
| 4490 | Megabalanus_volcano | Megabalanus | 0.15 | não | não |
| 6763 | Rhipicephalus_australis | Rhipicephalus | 0.15 | não | não |
| 6140 | Podospora_anserina | Podospora | 0.1467 | não | não |
| 317 | Amblyomma_americanum | Amblyomma | 0.1433 | não | não |
| 1708 | Chrysoporthe_deuterocubensis | Chrysoporthe | 0.1433 | não | não |
| 2722 | Eukaryota_sp_BB2 | Eukaryota | 0.1433 | não | não |
| 356 | Amolops_wuyiensis | Amolops | 0.14 | não | não |
| 474 | Anisandrus_dispar | Anisandrus | 0.14 | não | não |
| 2104 | Cyanoptyche_gloeocystis | Cyanoptyche | 0.14 | não | não |

## Limites

- **A classificação final é taxonomia onde ela existe e modelo no restante.** Os 79 morcegos vêm todos da lista taxonômica; o modelo não acrescentou nenhum.
- **Os 7122 organismos sem rótulo ficaram com classe 0 sem verificação taxonômica.** A decisão é só do modelo, e todas as pontuações são 0.
- **Os 300 negativos prováveis (249 na base e 51 no teste) não foram verificados**: foram sorteados entre gêneros fora da lista de mamíferos.
- **Ausência de candidatos não prova ausência de morcegos fora do MDD.** Um morcego cujo gênero não casou com a lista ficaria sem rótulo e, se o modelo não o reconhecesse, seria 0.
- Nomes de gênero antigos ou grafias diferentes podem não ter casado com a lista; os gêneros da lista sem correspondência em `Names.txt` estão em `results/generos_lista_sem_correspondencia.csv`.
- A contribuição do aprendizado de máquina está na simulação do oculto (Etapa 3: recuperação de 61 de 61 morcegos do oculto nas rodadas manuais, incluindo famílias sem representante na semente), na verificação do pool sem rótulo (aqui, com a segunda opinião) e na avaliação (Etapa 4). Na lista entregue, a contribuição direta do modelo é nula: nenhum positivo novo.

## Figuras

- `results/figuras/etapa5_fontes_rotulo.png`
- `results/figuras/etapa5_morcegos_por_familia.png`
- `results/figuras/etapa5_pontuacoes_sem_rotulo.png`
- `results/figuras/etapa5_mapa_pca.png`
- UMAP não gerado: `results/umap_cache.npy` não existe. O UMAP não é recalculado na Etapa 5 (a projeção já causou MemoryError na máquina da equipe).

## Como reproduzir

Na raiz do repositório, com o ambiente ativado, nesta ordem:

```bash
python -m src.exploracao
python -m src.supervisao
python -m src.experimento_etapa2
python -m src.enriquecimento --rodada 1   # e 2, 3, preenchendo revisao_manual.csv entre elas
python -m src.avaliacao_final --rodada-final 3 --avaliar-teste
python scripts/gerar_classes_finais.py --rodada-final 3 --base-lista 1
python -m src.relatorio_etapa5 --rodada-final 3
```

O teste congelado só é avaliado com `--avaliar-teste` (Etapa 4), e cada uso fica registrado em `results/etapa4_log_uso_teste.csv`. A Etapa 5 não reavalia o teste. Tudo usa `random_state=42`; repetir os comandos dá as mesmas saídas.
