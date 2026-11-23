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

**Objetivo:** entender o formato real dos arquivos e montar o ambiente.

**Fatos verificados nos arquivos enviados:**
- `Names.txt` e `wpc50.dat` têm **8426 linhas cada**, alinhadas pela posição (linha *i* de um corresponde à linha *i* do outro). `wpc50.dat` tem 50 colunas numéricas, sem valores ausentes, com média ≈ 0 e desvio ≈ 0,4.
- Os arquivos usam **quebra de linha CRLF** e o `Names.txt` tem **espaços à direita** em cada linha: é preciso fazer `strip()` ao ler.
- Cada nome está no formato `Genero_especie` (ex.: `Myotis_lucifugus`). Há **4848 gêneros distintos** e nenhum nome repetido. **Não há família, ordem nem qualquer hierarquia taxonômica.**
- Existem nomes fora do padrão: subespécies (`Rhinolophus_ferrumequinum_korai`), `sp.` (`Uropsilus_sp_1_FT_2014`), `nomen_nudum`, um híbrido de carpas e **um nome sem epíteto**. Tratar o gênero como "primeiro termo antes do `_`" funciona para todos, mas vale conferir esses casos.
- O arquivo **não está totalmente em ordem alfabética**: as 9 primeiras linhas (*Candida*..., *Cyprinus*...) saem do padrão. Não reordenar nada; usar sempre o número da linha original como índice.
- A base cobre **toda a árvore da vida** (fungos, peixes, insetos, plantas, aves etc.), então a maioria dos negativos é fácil. A dificuldade real está nos **mamíferos**.

**Tarefas:**
- [ ] Criar repositório (GitHub) e ambiente Python (`numpy`, `pandas`, `scikit-learn`, `matplotlib`, `umap-learn`, `imbalanced-learn`)
- [ ] Função de carga que remove `\r` e espaços, e verifica 8426 nomes × 8426 vetores
- [ ] Definir e documentar o índice base usado nas saídas (0 ou 1) e usar o mesmo em todo o projeto
- [ ] Extrair a coluna `genero` (primeiro termo do nome) e listar os casos fora do padrão
- [ ] Decidir sobre normalização (os dados já parecem centrados; avaliar se `StandardScaler` ajuda no SVM/k-NN)
- [ ] Primeira projeção (PCA/UMAP) para ver a estrutura geral

**Saída:** repositório pronto, função de carga, tabela `indice, nome, genero`.
**Critério de pronto:** dados carregados e alinhados; todos rodam o código na própria máquina.

---

### Etapa 1 — Supervisão inicial (sementes) — REVISADA

**Problema:** o `Names.txt` traz só o nome científico da espécie. Não há família nem ordem, então não dá para filtrar por "Vespertilionidae" ou "Chiroptera" diretamente. O **gênero**, porém, está embutido no nome (primeiro termo), e é a chave de ligação com fontes taxonômicas externas.

**Estratégia: supervisão por gênero, em dois níveis.**

1. **Positivos por gênero conhecido.**
   - Montar uma lista de gêneros da ordem *Chiroptera* a partir de fonte externa (Mammal Diversity Database, NCBI Taxonomy, ITIS ou GBIF; a ordem tem cerca de 200 gêneros).
   - Cruzar com a coluna `genero` do `Names.txt`. Só entra como positivo quem casa pelo gênero.
   - Teste preliminar já feito com uma lista de 60 gêneros: **74 instâncias de 27 gêneros** casaram (por exemplo *Myotis* com 28 espécies, *Rhinolophus* com 10, *Pteropus* com 4, *Artibeus*, *Carollia*, *Desmodus*, *Vampyrum*, *Tadarida*). O arquivo `sementes_positivas_iniciais.csv` traz essa lista. **Ela foi montada de memória e precisa ser conferida na fonte taxonômica pela equipe.**
   - Ampliar a lista de gêneros é parte do trabalho da Pessoa A: cada gênero novo achado na fonte pode trazer novos positivos.

2. **Negativos, com foco em mamíferos.**
   - Como o restante da base é de outros grupos, negativos aleatórios dão pouca informação. Priorizar **mamíferos não-morcegos** (roedores, primatas, carnívoros, ungulados, e especialmente insetívoros como musaranhos, toupeiras e ouriços), identificados por lista de gêneros de mamíferos da mesma fonte.
   - Completar com uma amostra aleatória de outros grupos (peixes, aves, insetos, plantas, fungos) para cobrir o resto da árvore.

**Validação importante (evitar vazamento por gênero):**
- Espécies do mesmo gênero são quase duplicatas no espaço de características (28 *Myotis*, por exemplo). Validação cruzada comum **infla o desempenho**, porque o modelo vê parentes próximos no treino e no teste.
- Usar **validação agrupada por gênero** (`GroupKFold` com `genero` como grupo).
- O **conjunto de teste verificado** deve ser formado por **gêneros inteiros reservados**, nunca usados no treino nem no enriquecimento. Incluir ao menos alguns gêneros de morcegos de famílias diferentes (ex.: um de Pteropodidae, um de Phyllostomidae, um de Vespertilionidae) para medir se o modelo generaliza para gêneros novos.

**Tarefas:**
- [ ] Obter e citar a lista de gêneros de *Chiroptera* (fonte e data de acesso)
- [ ] Cruzar a lista com a coluna `genero` e gerar `positivos.csv` (índice, nome, gênero, fonte)
- [ ] Obter lista de gêneros de mamíferos para montar os negativos difíceis
- [ ] Amostrar negativos de outros grupos para completar
- [ ] Separar e congelar o teste verificado por gêneros inteiros
- [ ] Registrar nomes ambíguos (subespécies, `sp.`, sinônimos) em uma tabela à parte, sem rotular até resolver

**Saída:** `positivos.csv`, `negativos.csv`, `teste_verificado.csv`, tabela de fontes.
**Critério de pronto:** sementes cobrem várias famílias (sabendo-as pela fonte externa, não pelo arquivo); teste separado por gêneros e congelado.

**Resultado de um teste rápido (para calibrar expectativa):** treinando Random Forest e SVM com os 74 positivos por gênero, o modelo, na validação cruzada simples, atingiu F1 de 0,93 e 0,95 para morcegos, e **apontou 5 candidatos novos de gêneros que não estavam na lista**: *Chalinolobus tuberculatus*, *Brachyphylla cavernarum*, *Lophostoma silvicolum*, *Tonatia saurophila* e *Rhinophylla pumilio* (todos conhecidos como morcegos, mas a equipe deve confirmar na fonte). Isso mostra que o ciclo de enriquecimento funciona. Esse F1 usou validação sem agrupar por gênero, então **está otimista**; o valor honesto vem da validação agrupada. Os falsos positivos mais próximos foram insetívoros como *Hylomys suillus* e *Uropsilus*, confirmando que os negativos difíceis devem ser mamíferos pequenos.

---

### Etapa 2 — Escolha e comparação de classificadores

**Objetivo:** escolher o classificador com base em números, não em preferência.

**Tarefas:**
- [ ] Implementar k-NN, SVM (RBF) e Random Forest com `class_weight='balanced'` quando aplicável
- [ ] Ajustar hiperparâmetros com validação cruzada estratificada (ex.: `k`, `C`, `gamma`, `n_estimators`)
- [ ] Comparar por recall (sensibilidade), precisão, F1 e acurácia
- [ ] Usar validação agrupada por gênero (`GroupKFold`) em todas as comparações
- [ ] Avaliar também um ensemble (votação) para usar como critério de confiança
- [ ] Justificar a escolha por escrito (isso entra na seção "Escolha do classificador")

**Saída:** tabela comparativa de modelos e classificador(es) escolhido(s).
**Critério de pronto:** escolha justificada com métricas e explicação de como o classificador será aplicado.

---

### Etapa 3 — Enriquecimento iterativo da base

**Objetivo:** expandir os positivos/negativos em ciclos controlados até estabilizar.

**Tarefas (a cada rodada):**
- [ ] Treinar com a base atual
- [ ] Predizer o restante e calcular probabilidades
- [ ] Selecionar apenas predições de alta confiança (ex.: p > 0,95, ou consenso do ensemble)
- [ ] **Conferir os nomes** das instâncias candidatas (validação humana via taxonomia) antes de incorporá-las
- [ ] Pesquisar manualmente as instâncias mais incertas (p ≈ 0,5), como *active learning*
- [ ] Opcional: complementar com propagação por vizinhança (*Label Spreading*) e balanceamento (SMOTE só no treino)
- [ ] Salvar rótulos da rodada em `data/rounds/` e registrar: nº de positivos, nº adicionados, nº corrigidos manualmente
- [ ] Parar quando uma rodada não adicionar novos positivos relevantes, ou quando restarem só casos ambíguos

**Saída:** histórico de rodadas (tabela e gráfico da evolução do nº de positivos), base final supervisionada.
**Critério de pronto:** processo documentado e reprodutível; critério de parada explícito.

---

### Etapa 4 — Modelo final e testes de performance

**Objetivo:** medir o desempenho de forma honesta.

**Tarefas:**
- [ ] Treinar o modelo final com a base enriquecida
- [ ] Avaliar no **conjunto de teste verificado, formado por gêneros inteiros reservados** (nunca usado no treino): sensibilidade, precisão, acurácia, F1 e matriz de confusão
- [ ] Reportar também validação cruzada estratificada no conjunto final
- [ ] Discutir limitações: acurácia sozinha engana com classes desbalanceadas; métricas sobre rótulos gerados pelo próprio modelo ficam infladas
- [ ] Analisar erros (falsos positivos e falsos negativos) olhando os nomes

**Saída:** tabela de métricas, matriz de confusão, análise de erros.
**Critério de pronto:** métricas calculadas em dados verificados, com discussão dos limites.

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
| **Pessoa B** | Modelagem e enriquecimento | Etapas 2 e 3 (comparação de classificadores, loop de self-training, active learning, balanceamento); código em `src/` |
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
| Formato dos arquivos (CRLF, espaços à direita, primeiras linhas fora de ordem) | Limpar na carga e nunca reordenar; usar o número da linha original |
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
