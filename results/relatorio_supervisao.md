# Relatório da supervisão inicial (Etapa 1, modo semente reduzida)

Lista taxonômica: 1363 gêneros; fonte(s): Mammal Diversity Database v2.5 (https://www.mammaldiversity.org/). Índices na base 0 (`BASE_INDICE`).

Parâmetros: `n_negativos=300`, `fracao_teste_negativos=0.2`, `fracao_semente=0.35`, `max_especies_teste=9`, `min_positivos_teste=5`, `random_state=42`.

## Avisos

_(nenhum)_

## Grupos

- `semente`: conhecido na rodada 0 (`semente_reduzida.csv` = `treino_inicial.csv`);
- `oculto`: tem gabarito, mas não entra no treino da rodada 0; pool onde o enriquecimento é medido (`src.validacao.avaliar_candidatos`);
- `teste`: gêneros inteiros reservados e bloqueados (`generos_bloqueados.csv`).

Células: instâncias (gêneros).

### Morcegos por família e grupo

Total: 79 instâncias de 32 gêneros.

| familia | semente | oculto | teste | total |
|---|---|---|---|---|
| Emballonuridae | 1 (1) | 0 | 0 | 1 (1) |
| Hipposideridae | 0 | 1 (1) | 0 | 1 (1) |
| Molossidae | 0 | 2 (1) | 0 | 2 (1) |
| Mormoopidae | 0 | 3 (1) | 0 | 3 (1) |
| Mystacinidae | 0 | 1 (1) | 0 | 1 (1) |
| Phyllostomidae | 4 (4) | 7 (6) | 1 (1) | 12 (11) |
| Pteropodidae | 4 (1) | 2 (2) | 1 (1) | 7 (4) |
| Rhinolophidae | 0 | 10 (1) | 0 | 10 (1) |
| Vespertilionidae | 4 (3) | 35 (5) | 3 (3) | 42 (11) |
| **total** | 13 (9) | 61 (18) | 5 (5) | 79 (32) |

### Negativos por ordem e grupo

Mamíferos não-morcegos: 921 instâncias de 428 gêneros. A linha "(negativo provável)" é a amostra aleatória sem gabarito.

| ordem | semente | oculto | teste | total |
|---|---|---|---|---|
| (negativo provável) | 249 (229) | 0 | 51 (48) | 300 (277) |
| Afrosoricida | 0 | 1 (1) | 2 (2) | 3 (3) |
| Artiodactyla | 75 (33) | 135 (61) | 30 (17) | 240 (111) |
| Carnivora | 40 (19) | 73 (35) | 22 (13) | 135 (67) |
| Cingulata | 2 (2) | 6 (4) | 12 (3) | 20 (9) |
| Dasyuromorphia | 3 (2) | 3 (3) | 1 (1) | 7 (6) |
| Dermoptera | 0 | 1 (1) | 0 | 1 (1) |
| Didelphimorphia | 1 (1) | 1 (1) | 3 (3) | 5 (5) |
| Diprotodontia | 4 (4) | 7 (6) | 3 (3) | 14 (13) |
| Eulipotyphla | 11 (6) | 27 (11) | 3 (3) | 41 (20) |
| Hyracoidea | 0 | 1 (1) | 1 (1) | 2 (2) |
| Lagomorpha | 3 (1) | 10 (1) | 1 (1) | 14 (3) |
| Macroscelidea | 1 (1) | 1 (1) | 0 | 2 (2) |
| Microbiotheria | 0 | 1 (1) | 0 | 1 (1) |
| Monotremata | 1 (1) | 1 (1) | 1 (1) | 3 (3) |
| Notoryctemorphia | 1 (1) | 0 | 0 | 1 (1) |
| Paucituberculata | 0 | 1 (1) | 1 (1) | 2 (2) |
| Peramelemorphia | 1 (1) | 2 (2) | 1 (1) | 4 (4) |
| Perissodactyla | 3 (2) | 13 (3) | 1 (1) | 17 (6) |
| Pholidota | 0 | 8 (1) | 0 | 8 (1) |
| Pilosa | 2 (1) | 4 (2) | 3 (2) | 9 (5) |
| Primates | 76 (19) | 94 (34) | 36 (15) | 206 (68) |
| Proboscidea | 0 | 2 (1) | 2 (1) | 4 (2) |
| Rodentia | 63 (27) | 99 (50) | 16 (12) | 178 (89) |
| Scandentia | 0 | 1 (1) | 0 | 1 (1) |
| Sirenia | 0 | 1 (1) | 1 (1) | 2 (2) |
| Tubulidentata | 0 | 1 (1) | 0 | 1 (1) |
| **total** | 536 (350) | 494 (225) | 191 (130) | 1221 (705) |

### Resumo por grupo

| grupo | positivos | negativos | generos |
|---|---|---|---|
| semente | 13 | 536 | 359 |
| oculto | 61 | 494 | 243 |
| teste | 5 | 184 | 135 |

## Teste verificado

189 instâncias de 135 gêneros inteiros (positivos: 5, negativos: 184). 7 nomes ambíguos de gêneros reservados ficaram fora do teste e do treino.

| origem | ordem | familia | instancias | generos |
|---|---|---|---|---|
| chiroptera_por_genero | Chiroptera | Phyllostomidae | 1 | 1 |
| chiroptera_por_genero | Chiroptera | Pteropodidae | 1 | 1 |
| chiroptera_por_genero | Chiroptera | Vespertilionidae | 3 | 3 |
| mamifero_por_genero | Afrosoricida | Chrysochloridae | 2 | 2 |
| mamifero_por_genero | Artiodactyla | Balaenidae | 1 | 1 |
| mamifero_por_genero | Artiodactyla | Bovidae | 13 | 6 |
| mamifero_por_genero | Artiodactyla | Cervidae | 7 | 3 |
| mamifero_por_genero | Artiodactyla | Delphinidae | 2 | 2 |
| mamifero_por_genero | Artiodactyla | Giraffidae | 1 | 1 |
| mamifero_por_genero | Artiodactyla | Lipotidae | 1 | 1 |
| mamifero_por_genero | Artiodactyla | Pontoporiidae | 1 | 1 |
| mamifero_por_genero | Artiodactyla | Tragulidae | 1 | 1 |
| mamifero_por_genero | Artiodactyla | Ziphiidae | 1 | 1 |
| mamifero_por_genero | Carnivora | Canidae | 2 | 2 |
| mamifero_por_genero | Carnivora | Eupleridae | 1 | 1 |
| mamifero_por_genero | Carnivora | Felidae | 6 | 3 |
| mamifero_por_genero | Carnivora | Herpestidae | 1 | 1 |
| mamifero_por_genero | Carnivora | Mustelidae | 2 | 2 |
| mamifero_por_genero | Carnivora | Otariidae | 1 | 1 |
| mamifero_por_genero | Carnivora | Phocidae | 8 | 3 |
| mamifero_por_genero | Cingulata | Chlamyphoridae | 1 | 1 |
| mamifero_por_genero | Cingulata | Dasypodidae | 7 | 1 |
| mamifero_por_genero | Cingulata | Tolypeutidae | 4 | 1 |
| mamifero_por_genero | Dasyuromorphia | Dasyuridae | 1 | 1 |
| mamifero_por_genero | Didelphimorphia | Didelphidae | 3 | 3 |
| mamifero_por_genero | Diprotodontia | Macropodidae | 1 | 1 |
| mamifero_por_genero | Diprotodontia | Petauridae | 1 | 1 |
| mamifero_por_genero | Diprotodontia | Vombatidae | 1 | 1 |
| mamifero_por_genero | Eulipotyphla | Talpidae | 3 | 3 |
| mamifero_por_genero | Hyracoidea | Procaviidae | 1 | 1 |
| mamifero_por_genero | Lagomorpha | Leporidae | 1 | 1 |
| mamifero_por_genero | Monotremata | Tachyglossidae | 1 | 1 |
| mamifero_por_genero | Paucituberculata | Caenolestidae | 1 | 1 |
| mamifero_por_genero | Peramelemorphia | Peramelidae | 1 | 1 |
| mamifero_por_genero | Perissodactyla | Tapiridae | 1 | 1 |
| mamifero_por_genero | Pilosa | Cyclopedidae | 1 | 1 |
| mamifero_por_genero | Pilosa | Myrmecophagidae | 2 | 1 |
| mamifero_por_genero | Primates | Atelidae | 1 | 1 |
| mamifero_por_genero | Primates | Callitrichidae | 1 | 1 |
| mamifero_por_genero | Primates | Cebidae | 2 | 1 |
| mamifero_por_genero | Primates | Cercopithecidae | 17 | 6 |
| mamifero_por_genero | Primates | Cheirogaleidae | 2 | 1 |
| mamifero_por_genero | Primates | Hylobatidae | 3 | 1 |
| mamifero_por_genero | Primates | Indriidae | 1 | 1 |
| mamifero_por_genero | Primates | Lemuridae | 1 | 1 |
| mamifero_por_genero | Primates | Lorisidae | 4 | 2 |
| mamifero_por_genero | Proboscidea | Elephantidae | 2 | 1 |
| mamifero_por_genero | Rodentia | Bathyergidae | 1 | 1 |
| mamifero_por_genero | Rodentia | Caviidae | 1 | 1 |
| mamifero_por_genero | Rodentia | Cricetidae | 2 | 2 |
| mamifero_por_genero | Rodentia | Echimyidae | 1 | 1 |
| mamifero_por_genero | Rodentia | Hystricidae | 1 | 1 |
| mamifero_por_genero | Rodentia | Muridae | 8 | 4 |
| mamifero_por_genero | Rodentia | Octodontidae | 1 | 1 |
| mamifero_por_genero | Rodentia | Sciuridae | 1 | 1 |
| mamifero_por_genero | Sirenia | Trichechidae | 1 | 1 |
| negativo_provavel | - | - | 51 | 48 |

Gêneros de morcego no teste: Anoura, Hypsugo, Lasiurus, Nyctalus, Rousettus.

## Gêneros bloqueados

135 gêneros (`data/seeds/generos_bloqueados.csv`), com 252 instâncias no Names.txt, das quais 107 sem rótulo. Todas ficam fora do treino e do enriquecimento (`src.supervisao.instancias_bloqueadas`).

| origem | generos | instancias |
|---|---|---|
| chiroptera_por_genero | 5 | 5 |
| mamifero_por_genero | 82 | 140 |
| negativo_provavel | 48 | 107 |

## Negativos prováveis

300 instâncias (pedido: 300) de 277 gêneros, amostradas entre as instâncias sem rótulo (random_state=42).

> **Atenção:** os negativos prováveis NÃO são verificados. A lista cobre só mamíferos; um mamífero ausente da lista (sinônimo, erro de grafia) pode estar entre eles. Conferir os nomes antes de confiar nas métricas.

## Instâncias sem rótulo

7426 instâncias sem rótulo pela lista (4 delas de gênero suspeito); 7126 continuam sem rótulo após a amostra de negativos prováveis.

## Casos ambíguos e suspeitos

388 nomes (detalhes em `data/seeds/casos_ambiguos.csv`).

| decisao | instancias |
|---|---|
| sem rotulo (genero fora da lista) | 302 |
| rotulado por genero; grupo semente | 29 |
| rotulado por genero; grupo oculto (so gabarito) | 28 |
| negativo provavel; grupo semente | 14 |
| excluido (genero reservado ao teste) | 7 |
| sem rotulo; genero bloqueado (teste) | 4 |
| sem rotulo (genero suspeito) | 4 |

## Correspondência entre a lista e o Names.txt

Gêneros da lista ausentes do Names.txt: 903 (lista completa em `results/generos_lista_sem_correspondencia.csv`).

| ordem | generos |
|---|---|
| Rodentia | 451 |
| Chiroptera | 206 |
| Carnivora | 62 |
| Eulipotyphla | 42 |
| Artiodactyla | 28 |
| Diprotodontia | 28 |
| Afrosoricida | 17 |
| Primates | 16 |
| Dasyuromorphia | 13 |
| Didelphimorphia | 13 |
| Lagomorpha | 9 |
| Macroscelidea | 4 |
| Peramelemorphia | 4 |
| Scandentia | 3 |
| Pholidota | 2 |
| Cingulata | 1 |
| Dermoptera | 1 |
| Hyracoidea | 1 |
| Paucituberculata | 1 |
| Sirenia | 1 |

Gêneros de Chiroptera da lista ausentes do Names.txt:

Acerodon, Aethalops, Afronycteris, Afropipistrellus, Alionoctula, Alionycteris, Ametrida, Amorphochilus, Anthops, Antrozous, Aproteles, Ardops, Arielulus, Ariteus, Asellia, Aselliscus, Austronomus, Baeodon, Balantiopteryx, Balionycteris, Barbastella, Bauerus, Boneia, Cabreramops, Cardioderma, Casinycteris, Cassistrellus, Centronycteris, Centurio, Cheiromeles, Chilonatalus, Chiroderma, Chironax, Choeroniscus, Choeronycteris, Chrotopterus, Cistugo, Cloeotis, Cnephaeus, Coelops, Coleura, Cormura, Craseonycteris, Cynomops, Cyttarops, Dermanura, Desmalopex, Diaemus, Diclidurus, Diphylla, Dobsonia, Doryrhina, Dryadonycteris, Dyacopterus, Ectophylla, Eidolon, Emballonura, Enchisthenes, Eonycteris, Epomops, Erophylla, Euderma, Eudiscoderma, Eudiscopus, Eumops, Falsistrellus, Furipterus, Gardnerycteris, Glauconycteris, Glischropus, Glossophaga, Glyphonycteris, Haplonycteris, Harpiocephalus, Harpiola, Harpyionycteris, Hesperoptenus, Histiotus, Hsunycteris, Hylonycteris, Hypsignathus, Ia, Idionycteris, Kerivoula, Laephotis, Lampronycteris, Lasionycteris, Latidens, Lavia, Leptonycteris, Lichonycteris, Lionycteris, Lonchophylla, Lonchorhina, Lyroderma, Macroderma, Macroglossus, Macronycteris, Macrophyllum, Macrotus, Megaderma, Megaerops, Megaloglossus, Melonycteris, Mesophylla, Micronomus, Mimetillus, Mimon, Miniopterus, Mirimiri, Mirostrellus, Molossops, Molossus, Monophyllus, Mops, Mormoops, Mormopterus, Mosia, Musonycteris, Myonycteris, Myopterus, Myzopoda, Nanonycteris, Natalus, Neoeptesicus, Neonycteris, Neoplatymops, Neopteryx, Neoromicia, Nesonycteris, Noctilio, Notopteris, Nycteris, Nycticeinops, Nycticeius, Nyctiellus, Nyctimene, Nyctinomops, Nyctinomus, Nyctophilus, Otomops, Otonycteris, Otopteropus, Ozimops, Paranyctimene, Parastrellus, Paratriaenops, Paremballonura, Penthetor, Perimyotis, Peropteryx, Pharotis, Philetor, Phoniscus, Phylloderma, Phyllonycteris, Phyllops, Phyllostomus, Pilonycteris, Platalina, Platymops, Platyrrhinus, Plerotes, Promops, Pseudoromicia, Ptenochirus, Pteralopex, Pygoderma, Rhinonicteris, Rhinopoma, Rhogeessa, Rhynchonycteris, Rhyneptesicus, Saccolaimus, Sauromys, Scleronycteris, Scoteanax, Scotoecus, Scotomanes, Scotonycteris, Scotophilus, Scotorepens, Scotozous, Setirostris, Sphaerias, Sphaeronycteris, Stenoderma, Stenonycteris, Styloctenium, Submyotodon, Syconycteris, Taphozous, Thoopterus, Thyroptera, Tomopeas, Trachops, Triaenops, Trinycteris, Tylonycteris, Uroderma, Vampyressa, Vampyriscus, Vampyrodes, Vansonia, Vespadelus, Xeronycteris


No sentido Names.txt → lista: 4384 gêneros sem correspondência (não dá para saber quais são mamíferos sem a lista); 0 gêneros casados via sinônimo (0 instâncias).
