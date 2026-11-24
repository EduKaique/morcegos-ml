# Plano de Execução — Reconhecimento de Padrões: "É morcego?"

Disciplina: Inteligência Computacional Aplicada I (DS803) — TADS/UFPR

## 1. Visão geral

**Objetivo:** classificar as 8426 instâncias de `Wpc50.dat` (50 características de genomas mitocondriais) em `1 = Morcego` ou `0 = Outro`, usando `Names.txt` (nomes científicos) como apoio para a supervisão, com enriquecimento iterativo da base.

**Entregáveis finais:**

- [ ] Vetor com as 8426 classes (1/0), em CSV/TXT
- [ ] Lista de índices (número de ordem) das instâncias positivas
- [ ] Código-fonte documentado
- [ ] Relatório com: Resumo, Apresentação e Introdução, Estratégias de classificação, Escolha do classificador, Testes de performance, Conclusão
- [ ] Apresentação pública no último dia da disciplina

**Pipeline resumido:**

```
Sementes (pesquisa) → treino → predição no restante → filtro por confiança
→ conferência pelos nomes → adiciona à base → repete até estabilizar
→ modelo final + métricas no teste verificado → exporta 8426 classes
```

> **Atenção:** o enunciado diz que a equipe pode ter *até duas pessoas*. Como somos três, vale confirmar com o professor (Raittz) antes de seguir.

---

## 2. Estrutura sugerida do repositório

```
morcegos/
├── data/
│   ├── raw/                 # Wpc50.dat, Names.txt (não alterar)
│   ├── seeds/               # positivos.csv, negativos.csv, teste_verificado.csv
│   └── rounds/              # rótulos de cada rodada (rodada_01.csv, ...)
├── notebooks/
├── src/
│   ├── carregar_dados.py
│   ├── sementes.py
│   ├── self_training.py
│   ├── avaliacao.py
│   └── visualizacao.py
├── results/
│   ├── classes_finais.csv   # 8426 linhas (índice, nome, classe)
│   ├── indices_positivos.txt
│   └── figuras/
├── relatorio/
└── README.md
```

---

## 3. Plano por etapa

### Etapa 0 — Preparação e exploração dos dados

**Status: concluída** (commit inicial na `main`; código em `src/carregar_dados.py` e `src/exploracao.py`).

**Objetivo:** entender o formato real dos arquivos e montar o ambiente.

**Fatos verificados nos arquivos enviados:**
- `Names.txt` e `wpc50.dat` têm **8426 linhas cada**, alinhadas pela posição (linha *i* de um corresponde à linha *i* do outro). `wpc50.dat` tem 50 colunas numéricas, sem valores ausentes, com média global ≈ −0,04 e desvio global ≈ 0,40. **As colunas não estão na mesma escala**: a coluna 0 tem média ≈ −1,80 e a coluna 1 ≈ −0,28, então a normalização precisa ser testada (k-NN e SVM, baseados em distância, são sensíveis; Random Forest não).
- Os arquivos usam **quebra de linha CRLF** e o `Names.txt` tem **espaços à direita** em cada linha: é preciso fazer `strip()` ao ler.
- Cada nome está no formato `Genero_especie` (ex.: `Myotis_lucifugus`). Há **4847 gêneros distintos** e nenhum nome repetido. **Não há família, ordem nem qualquer hierarquia taxonômica.**
- Há **389 nomes fora do padrão** (388 tratados como ambíguos; *Crocodylus niloticus* só tem espaço no lugar de `_`): subespécies e nomes com mais de dois termos (387), `sp` (108, ex.: `Uropsilus_sp_1_FT_2014`), híbridos (64) e `nomen_nudum` (3). Dois nomes usam **espaço no lugar de `_`** (`Calosoma sp. BYU_CO241` e `Crocodylus niloticus`), por isso o gênero é extraído separando por `_` **ou espaço**. Não existe nome sem epíteto.
- As 4 últimas linhas (8422 a 8425) começam com `diploid_`, `triploid_` e `natural_gynogenetic_`: são híbridos de peixes, e o primeiro termo **não é um gênero de verdade**. Esses casos ficam marcados como "gênero suspeito" e **não devem ser rotulados por gênero** na Etapa 1.
- As 9 primeiras linhas (8 *Candida* e um híbrido de carpas) saem da ordem do resto do arquivo, que segue ordem ASCII (maiúsculas antes de minúsculas), o que causa quebras aparentes no meio. Não reordenar nada; usar sempre o número da linha original como índice (base 0 no código, `BASE_INDICE = 0`).
- A base cobre **toda a árvore da vida** (fungos, peixes, insetos, plantas, aves etc.), então a maioria dos negativos é fácil. A dificuldade real está nos **mamíferos**.

**Tarefas:**
- [ ] Criar repositório (GitHub) e ambiente Python (`numpy`, `pandas`, `scikit-learn`, `matplotlib`, `umap-learn`, `imbalanced-learn`)
- [ ] Função de carga que remove `\r` e espaços, e verifica 8426 nomes × 8426 vetores
- [ ] Definir e documentar o índice base usado nas saídas (0 ou 1) e usar o mesmo em todo o projeto
- [ ] Extrair a coluna `genero` (primeiro termo do nome) e listar os casos fora do padrão
- [ ] Decidir sobre normalização: comparar com e sem `StandardScaler` na Etapa 2 (escalas diferentes entre colunas)
- [ ] Primeira projeção (PCA/UMAP) para ver a estrutura geral

**Saída:** repositório pronto, função de carga, tabela `indice, nome, genero`.
**Critério de pronto:** dados carregados e alinhados; todos rodam o código na própria máquina.

---

### Etapa 1 — Supervisão inicial (semente reduzida e gabarito)

**Status: implementada** (branch `etapa-1-supervisao`; `src/supervisao.py`, `src/validacao.py`). A lista taxonômica real é o MDD v2.5, acessada em 2026-11-23, e está em `data/external/generos_mamiferos.csv` (1363 gêneros, 27 ordens, 238 gêneros de *Chiroptera*).

**Problema original:** o `Names.txt` só traz `Genero_especie`, sem família nem ordem. O gênero é a ponte com a taxonomia externa.

**Decisão de método:** se todos os morcegos entrassem rotulados por regra, o classificador não teria nada a descobrir, e o enunciado pede prospecção. Por isso a taxonomia completa é usada como **gabarito** (conferir candidatos, medir métricas e fechar a entrega), e o treino começa com uma **semente reduzida**.

**Grupos (por gênero, `random_state=42`):**

| Grupo | Morcegos (gêneros) | Negativos | Papel |
|---|---|---|---|
| semente | 13 (9) | 536 (287 mamíferos + 249 negativos prováveis) | único grupo usado no treino da rodada 0 |
| oculto | 61 (18) | 494 | pool onde medimos a prospecção; tem gabarito, mas não entra no treino |
| teste | 5 (5): *Anoura*, *Rousettus*, *Hypsugo*, *Lasiurus*, *Nyctalus* | 184 (133 mamíferos + 51 negativos prováveis) | verificado, congelado por SHA256 (início `68bc13a7`) |

**Regras já implementadas:** gêneros com 10 ou mais espécies (como *Myotis*) nunca vão para o teste; nomes ambíguos e de gênero suspeito ficam fora do teste; os 135 gêneros do teste ficam **bloqueados** (252 instâncias no `Names.txt`, 107 sem rótulo) e não podem ser candidatos nem entrar no treino; mudar `--fracao-semente` não muda o teste nem o hash.

**Arquivos em `data/seeds/`:** `gabarito.csv`, `semente_reduzida.csv`, `teste_verificado.csv` (+ `.sha256`), `generos_bloqueados.csv`, `positivos.csv`, `negativos.csv`, `treino_inicial.csv`, `casos_ambiguos.csv`. **Treinar só com `semente_reduzida.csv`.**

**Pontos de atenção descobertos nesta etapa:**
- A semente tem só **13 positivos de 9 gêneros**: as métricas de validação serão ruidosas. Usar validação agrupada repetida e reportar desvio.
- O **oculto concentra 62% dos positivos em dois gêneros** (*Myotis* com 28 e *Rhinolophus* com 10). Acertar ou errar um deles muda o recall de forma abrupta. Reportar recall **por gênero e por família**, e não só o agregado.
- Cinco famílias do oculto **não têm representante na semente**: Rhinolophidae, Hipposideridae, Molossidae, Mormoopidae e Mystacinidae. Isso é o ponto difícil da prospecção e deve ser discutido no relatório.
- O teste tem **só 5 positivos**: acertar 5 de 5 ainda dá um intervalo de confiança de 95% de cerca de 57% a 100% (Wilson). Reportar sempre com intervalo.
- Os **negativos prováveis** (249 na semente, 51 no teste) são amostras aleatórias de outros grupos e **não foram verificados** taxonomicamente. Conferir uma amostra.
- As 107 instâncias bloqueadas sem rótulo não têm gabarito: na entrega final recebem a predição do modelo final (são gêneros de negativos prováveis), com conferência manual.

**Tarefas restantes:**
- [ ] Revisão por outra pessoa: preencher "Quem conferiu" em `docs/fontes_taxonomicas.md` após checar uma amostra (10 gêneros de morcego e 10 de insetívoros contra o site do MDD)
- [ ] Conferir manualmente os gêneros dos 51 negativos prováveis do teste
- [ ] Commitar `data/seeds/` (o congelamento só vale para a equipe se estiver no git) e abrir pull request
- [ ] Sensibilidade (opcional): repetir com `--fracao-semente` maior e comparar a recuperação do oculto

**Saída:** arquivos de `data/seeds/`, `results/relatorio_supervisao.md`.
**Critério de pronto:** pull request aprovado, "Quem conferiu" preenchido, teste congelado no git.

---

### Etapa 2 — Comparação de classificadores (k-NN, SVM RBF e Random Forest)

**Decisão da equipe:** comparar **três classificadores** (k-NN, SVM com kernel RBF e Random Forest), escolher **um** com base nos números e justificar por escrito. Os três têm lógicas diferentes: o k-NN se apoia na proximidade (espécies filogeneticamente próximas ficam próximas no espaço das 50 características), o SVM busca uma fronteira em um espaço transformado pelo kernel, e o Random Forest combina árvores e dá importância às características.

**Objetivo:** escolher o classificador e sua configuração por números, treinando só com a **semente reduzida**, e fundamentar a seção "Escolha do classificador".

**Desenho da avaliação:**
- **Seleção** por validação agrupada por gênero (`StratifiedGroupKFold`), **repetida com várias sementes** (por exemplo, 10), sobre `semente_reduzida.csv`, usando **as mesmas dobras para os três modelos**.
- **Diagnóstico** no grupo oculto: recall por gênero e família e falsos positivos entre mamíferos, para o melhor de cada modelo. O oculto **não** é usado para escolher hiperparâmetros nem modelo.
- O teste congelado não é tocado nesta etapa.

**Configurações a variar:**
- **k-NN:** `k` pequeno (1, 3, 5, 7), votos uniformes ou por distância, métrica (euclidiana, Manhattan, cosseno). Pontuação: fração ponderada de vizinhos positivos, ou razão entre a distância ao positivo e ao negativo mais próximos (o k-NN não tem `class_weight`).
- **SVM RBF:** `C` e `gamma` em grade pequena, `class_weight='balanced'`. Usar o valor da função de decisão como pontuação (a calibração de probabilidades com 13 positivos é instável).
- **Random Forest:** `n_estimators` e `max_depth`, `class_weight='balanced'`. Pontuação: probabilidade prevista; guardar também a importância das características.
- **Escala:** cada modelo com e sem `StandardScaler`, ajustado dentro de cada dobra. As colunas têm escalas diferentes: k-NN e SVM devem ser sensíveis; Random Forest não deve mudar.

**Tarefas:**
- [ ] Implementar os três modelos como `Pipeline`, com a mesma interface de pontuação
- [ ] Métricas: precisão média (área sob a curva precisão-recall), F1 no melhor limiar, recall com precisão ≥ 0,95, além de sensibilidade, precisão e acurácia
- [ ] Escolher o classificador: melhor precisão média; se a diferença for menor que 1 desvio-padrão (provável com tão poucos positivos), decidir por simplicidade e interpretabilidade e dizer isso no relatório
- [ ] Definir o **limiar de confiança** do modelo escolhido para o enriquecimento
- [ ] Diagnóstico no oculto por gênero e família; para o k-NN, listar os vizinhos mais próximos de cada morcego do oculto
- [ ] Escrever a justificativa da escolha

**Saída:** `results/etapa2_comparacao.csv`, figuras (curvas precisão-recall dos três modelos, com e sem escala, recall por família no oculto) e relatório.
**Critério de pronto:** classificador escolhido e justificado com números e desvio entre repetições, com explicação de como será aplicado.

---

### Etapa 3 — Enriquecimento iterativo da base

**Objetivo:** expandir os positivos e negativos em ciclos controlados, medindo contra o gabarito quantos morcegos do oculto são recuperados.

**Tarefas (a cada rodada):**
- [ ] Treinar com a base atual (rodada 0 = `semente_reduzida.csv`)
- [ ] Predizer o restante e calcular probabilidades; **excluir os candidatos bloqueados** (gêneros do teste)
- [ ] Selecionar predições de alta confiança (limiar da Etapa 2)
- [ ] **Conferir os nomes** das candidatas (validação humana via taxonomia/gabarito) antes de incorporá-las
- [ ] Pesquisar manualmente as mais incertas (p ≈ 0,5), como *active learning*, em especial para as **famílias sem representante na semente**
- [ ] Opcional: propagação por vizinhança (*Label Spreading*) e balanceamento (SMOTE só no treino)
- [ ] Medir com `avaliar_candidatos`: precisão e recall acumulados no oculto, por gênero e por família, e candidatos fora do gabarito
- [ ] Salvar rótulos em `data/rounds/` e registrar: nº de positivos, adicionados, corrigidos manualmente
- [ ] Parar quando uma rodada não adicionar positivos relevantes, ou restarem só casos ambíguos

**Pontos de atenção:** candidatos fora do gabarito (gêneros que o MDD não cobre ou nomes antigos) são justamente onde pode haver morcegos escondidos; investigá-los na taxonomia. Gêneros do teste ficam bloqueados.

**Saída:** histórico de rodadas (tabela e gráfico), base final supervisionada.
**Critério de pronto:** processo reprodutível; critério de parada explícito.

---

### Etapa 4 — Modelo final e testes de performance

**Objetivo:** medir o desempenho de forma honesta, sabendo que o teste tem poucos positivos.

**Tarefas:**
- [ ] Treinar o modelo final com a base enriquecida
- [ ] Avaliar no **teste congelado** (5 positivos de 5 gêneros e 184 negativos): sensibilidade, precisão, acurácia, F1 e matriz de confusão, **com intervalos de confiança de 95% (Wilson)**
- [ ] Complemento mais estável: validação agrupada por gênero (ou *leave-one-genus-out*) sobre **todo o gabarito** (79 positivos), apenas como medida complementar
- [ ] Reportar a recuperação do oculto por família (resultado principal da prospecção)
- [ ] Discutir limitações: acurácia sozinha engana com classes desbalanceadas; rótulos gerados pelo próprio modelo inflam métricas; negativos prováveis não verificados
- [ ] Analisar erros (falsos positivos e falsos negativos) olhando os nomes

**Saída:** tabela de métricas com intervalos, matriz de confusão, análise de erros.
**Critério de pronto:** métricas em dados verificados, com discussão dos limites.

---

### Etapa 5 — Geração dos resultados finais

**Objetivo:** produzir os arquivos exigidos pelo enunciado (item c).

**Tarefas:**
- [ ] Gerar `classes_finais.csv` com as 8426 linhas (índice, nome científico, classe 1/0)
- [ ] Gerar `indices_positivos.txt` com os números de ordem dos morcegos
- [ ] Conferir: 8426 linhas, somente 0/1, contagem de positivos coerente com a lista de índices
- [ ] Revisão cruzada: outra pessoa confere uma amostra dos positivos e dos negativos pelo nome

**Saída:** arquivos finais em `results/`.
**Critério de pronto:** arquivos validados por script e por revisão manual.

---

### Etapa 6 — Visualizações

**Objetivo:** materiais visuais para o relatório e a apresentação.

**Tarefas:**
- [ ] Projeção PCA/t-SNE/UMAP colorida por classe final
- [ ] Projeção colorida pela rodada em que cada instância foi rotulada
- [ ] Gráfico da evolução de positivos por rodada
- [ ] Matriz de confusão e comparação de modelos

**Saída:** figuras em `results/figuras/`.

---

### Etapa 7 — Relatório

**Estrutura exigida:**

1. **Resumo** (escrever por último)
2. **Apresentação e introdução:** problema, dados, motivação
3. **Estratégias utilizadas para a classificação:** supervisão inicial, enriquecimento, critérios de confiança, validação humana
4. **Escolha do classificador:** comparação e justificativa
5. **Testes de performance:** métricas e análise de erros
6. **Conclusão:** resultados, limitações, trabalhos futuros

**Tarefas:**
- [ ] Cada pessoa redige as seções da sua área (ver divisão abaixo)
- [ ] Uma pessoa faz a revisão final de coerência e formatação
- [ ] Conferir que o relatório cita as fontes taxonômicas

---

### Etapa 8 — Apresentação pública

**Tarefas:**
- [ ] Montar slides (sugestão: 10 a 12 slides): problema, supervisão, enriquecimento (com a visualização por rodada), classificador, métricas, conclusão
- [ ] Definir quem fala em cada parte (todos devem falar)
- [ ] Ensaio completo com cronômetro
- [ ] Preparar respostas para perguntas prováveis: *por que esse limiar de confiança? como evitou vazamento no teste? como tratou falsos positivos?*

---

## 4. Divisão de tarefas (3 pessoas)

> Substituir **Pessoa A/B/C** pelos nomes da equipe. Cada pessoa é responsável por uma área, mas todos revisam o trabalho dos outros nos pontos de integração.

| Pessoa | Papel | Responsabilidades principais |
|---|---|---|
| **Pessoa A** | Dados e supervisão | Etapas 0 e 1 (carga, exploração, lista de gêneros de *Chiroptera* e de mamíferos em fonte externa, sementes por gênero, teste verificado por gêneros); conferência por nomes na Etapa 3; Etapa 5 (arquivos finais) |
| **Pessoa B** | Modelagem e enriquecimento | Etapas 2 e 3 (comparação de classificadores, loop de enriquecimento, active learning, balanceamento); código em `src/` |
| **Pessoa C** | Avaliação, visualização e comunicação | Etapas 4 e 6 (métricas, matriz de confusão, gráficos); coordenação do relatório (Etapa 7) e dos slides (Etapa 8) |

### Responsabilidade por seção do relatório

| Seção | Responsável principal | Revisor |
|---|---|---|
| Resumo | Pessoa C | Todos |
| Apresentação e introdução | Pessoa A | Pessoa C |
| Estratégias de classificação | Pessoa B (enriquecimento) e Pessoa A (supervisão) | Pessoa C |
| Escolha do classificador | Pessoa B | Pessoa A |
| Testes de performance | Pessoa C | Pessoa B |
| Conclusão | Pessoa C | Todos |

### Pontos de integração (reuniões curtas)

1. **Fim da Etapa 1:** A entrega as sementes e o teste verificado; B e C validam o formato.
2. **Fim da Etapa 2:** B apresenta a comparação de modelos; o grupo decide o classificador.
3. **A cada rodada da Etapa 3:** A e B conferem juntos os candidatos novos pelos nomes.
4. **Fim da Etapa 4:** C apresenta as métricas; o grupo decide se precisa de mais uma rodada.
5. **Antes da entrega:** ensaio da apresentação com todos.

### Regras de trabalho em equipe

- Um branch por pessoa e *pull requests* revisados por pelo menos uma outra pessoa
- Semente aleatória fixa (`random_state`) para reprodutibilidade
- O teste verificado é congelado e só A pode alterá-lo, com aviso ao grupo
- Decisões de rotulagem registradas na tabela de fontes, não só em conversa

---

## 5. Cronograma sugerido

Ajustar às datas reais da disciplina. Exemplo em semanas até a apresentação:

| Semana | Atividades | Responsáveis |
|---|---|---|
| 1 | Etapa 0 (preparação) e início da Etapa 1 | A (lidera), B e C apoiam |
| 2 | Fim da Etapa 1; Etapa 2 (classificadores) | A e B |
| 3 | Etapa 3 (rodadas de enriquecimento 1 a 3) | B e A |
| 4 | Etapa 3 (rodadas finais), Etapa 4 (métricas), Etapa 5 (arquivos) | B, C, A |
| 5 | Etapa 6 (figuras), Etapa 7 (relatório) | C (lidera), todos |
| 6 | Etapa 8 (slides e ensaio), revisão final e entrega | Todos |

---

## 6. Riscos e mitigação

| Risco | Mitigação |
|---|---|
| `Names.txt` sem família/ordem; só há o gênero embutido no nome | Usar lista externa de gêneros (MDD/NCBI/GBIF) e cruzar pela coluna `genero` |
| Sementes pouco diversas (modelo só reconhece uma família) | Cobrir várias famílias de morcegos, conferidas pela fonte externa |
| Vazamento por gênero (espécies congenéricas são quase duplicatas) | `GroupKFold` por gênero e teste com gêneros inteiros reservados |
| Lista de gêneros montada de memória pode ter erros | Conferir cada gênero na fonte e registrar data de acesso |
| Propagação de erros no self-training | Limiar alto de confiança, consenso de modelos e conferência dos nomes a cada rodada |
| Métricas infladas por vazamento | Teste verificado congelado e fora do enriquecimento |
| Classes muito desbalanceadas | `class_weight`, SMOTE só no treino, foco em recall/precisão/F1 |
| Nomes ambíguos ou sinônimos taxonômicos | Consultar NCBI Taxonomy; registrar casos duvidosos |
| Formato dos arquivos (CRLF, espaços à direita, dois nomes com espaço no lugar de `_`, 4 nomes com gênero suspeito) | Limpar na carga, extrair o gênero separando por `_` ou espaço, nunca reordenar e usar o número da linha original |
| Colunas de `wpc50.dat` em escalas diferentes | Testar com e sem normalização na Etapa 2 |
| Nomes antigos de gêneros que não casam com a lista atual | Tabela de sinônimos e conferência dos gêneros sem correspondência |
| Equipe de 3 acima do limite do enunciado | Confirmar com o professor logo no início |

---

## 7. Checklist final de entrega

- [ ] `classes_finais.csv` com 8426 linhas (1/0)
- [ ] `indices_positivos.txt`
- [ ] Código organizado e comentado, com README de execução
- [ ] Relatório com todas as seções exigidas
- [ ] Métricas (sensibilidade, precisão, acurácia) no conjunto verificado
- [ ] Slides prontos e ensaiados
- [ ] Fontes taxonômicas citadas
