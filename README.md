# É morcego? — Reconhecimento de padrões em genomas mitocondriais

Trabalho da disciplina DS803 — Inteligência Computacional Aplicada I (TADS/UFPR).

## Objetivo

Classificar as 8426 instâncias de `data/raw/wpc50.dat` (50 características de
genomas mitocondriais) em `1 = morcego` ou `0 = outro`, usando os nomes
científicos de `data/raw/Names.txt` para supervisionar. A abordagem é
semi-supervisionada, com enriquecimento iterativo da base. O plano completo está
em [`docs/plano_execucao_morcegos.md`](docs/plano_execucao_morcegos.md) e o
enunciado em `docs/Trabalho_morcegos.pdf`.

## Ambiente

Requer Python 3.12.

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Como rodar

Sempre a partir da raiz do repositório, com o `.venv` ativo:

```bash
python -m src.exploracao   # estatísticas, nomes fora do padrão e figuras
pytest                     # testes (usam a lista real; o fixture só se ela estiver vazia)
```

A exploração grava `results/estatisticas_descritivas.csv`,
`results/nomes_fora_do_padrao.csv` e as figuras `pca_2d.png`, `umap_2d.png` e
`hist_especies_por_genero.png` em `results/figuras/`. As figuras não são
versionadas, porque o script as gera de novo.

## Como gerar a supervisão (Etapa 1)

1. Coloque a lista de gêneros de mamíferos em
   `data/external/generos_mamiferos.csv` e, se houver, os sinônimos em
   `data/external/sinonimos_generos.csv` (formatos em
   [`data/external/README.md`](data/external/README.md)). Registre as fontes em
   [`docs/fontes_taxonomicas.md`](docs/fontes_taxonomicas.md).
2. Rode:

   ```bash
   python -m src.supervisao                        # padrões abaixo
   python -m src.supervisao --n-negativos 500
   python -m src.supervisao --fracao-semente 0.5   # semente maior (não muda o teste)
   ```

   | Opção | Padrão | Efeito |
   |---|---|---|
   | `--n-negativos` | 300 | tamanho da amostra de negativos prováveis |
   | `--fracao-semente` | 0.35 | fração dos gêneros fora do teste que entram na semente |
   | `--max-especies-teste` | 9 | gêneros com mais instâncias que isso (ex.: *Myotis*) nunca vão para o teste |
   | `--min-positivos-teste` | 5 | completa o teste com gêneros de morcego até esse número de positivos |
   | `--sobrescrever-teste` | — | permite substituir o teste congelado (só com aval da equipe) |

   Sem a lista (só o cabeçalho), o script para com uma mensagem e não grava nada.
3. Leia `results/relatorio_supervisao.md` (inclusive a seção **Avisos**) e
   revise `data/seeds/casos_ambiguos.csv`.

### Modo semente reduzida: gabarito, semente, oculto e teste

Se todos os mamíferos rotulados pela taxonomia entrassem no treino, o
enriquecimento da Etapa 3 não teria nada para descobrir. Por isso a taxonomia
completa é o **gabarito**, e cada gênero rotulado cai em um de três grupos
(por gênero inteiro, nunca dividido):

| Grupo | O que é | Pode treinar? |
|---|---|---|
| `semente` | ~35% dos gêneros fora do teste (morcegos estratificados por família, outros mamíferos por ordem, sempre com Eulipotyphla), mais os negativos prováveis fora do teste | sim, é a rodada 0 |
| `oculto` | o resto dos gêneros com gabarito | **não** na rodada 0; só entram se forem descobertos pelo enriquecimento |
| `teste` | gêneros inteiros reservados ao teste verificado | **nunca** (nem no enriquecimento) |

- O **gabarito** (`gabarito.csv`) serve para conferir candidatos e medir
  métricas. **Nunca use como treino fora do grupo `semente`.**
- `src.validacao.avaliar_candidatos(candidatos_idx, gabarito)` mede a
  prospecção: precisão e recall acumulados dos candidatos adicionados no pool
  oculto, e quantos eram morcegos reais. Candidato do grupo `teste` gera erro.
- `src.supervisao.instancias_bloqueadas(df)` devolve os índices de **todas** as
  instâncias dos gêneros do teste no `Names.txt`, inclusive as sem rótulo. A
  Etapa 3 deve excluí-las dos candidatos e do treino; na entrega final elas
  recebem o rótulo do gabarito.
- Mudar `--fracao-semente` não altera o teste (o sorteio usa outro gerador).

Arquivos gerados em `data/seeds/`:

| Arquivo | Conteúdo |
|---|---|
| `gabarito.csv` | todas as instâncias com rótulo taxonômico: `indice, nome, genero, genero_atual, ordem, familia, rotulo_gabarito, grupo` |
| `semente_reduzida.csv` | só o que é conhecido na rodada 0 (grupo `semente`), com `rotulo` e `origem` |
| `treino_inicial.csv` | igual a `semente_reduzida.csv` |
| `teste_verificado.csv` | gêneros inteiros reservados, sem nomes ambíguos |
| `teste_verificado.sha256` | hash que congela o teste |
| `generos_bloqueados.csv` | todos os gêneros do teste, com nº de instâncias no Names.txt e no teste |
| `positivos.csv` | todos os morcegos rotulados pelo gênero, com a coluna `grupo` |
| `negativos.csv` | mamíferos não-morcegos e negativos prováveis, com a coluna `grupo` |
| `casos_ambiguos.csv` | nomes ambíguos e suspeitos, a decisão tomada e uma coluna `observacao` para a equipe |

Colunas comuns: `indice` (base 0), `nome`, `genero` (como está no `Names.txt`),
`genero_atual` (após sinônimos; é o grupo usado na validação), `ordem`,
`familia`, `rotulo` (1 morcego, 0 outro) e `origem`.

- **`positivos.csv`/`negativos.csv` incluem o oculto e o teste.** Para treinar,
  use `semente_reduzida.csv` (ou filtre `grupo == "semente"`).
- **Teste congelado:** se o teste já existe e uma nova execução geraria outro
  conteúdo, o script para. Só com o aval da equipe use `--sobrescrever-teste`,
  e registre a mudança em `docs/fontes_taxonomicas.md`.
- **Negativos prováveis não são verificados:** são instâncias de gêneros fora da
  lista de mamíferos; um mamífero ausente da lista pode estar entre eles. Eles
  não fazem parte do gabarito.
- **Rótulos do teste verificado na entrega final:** como foram verificados por
  taxonomia, esses rótulos também são usados diretamente no arquivo final das
  8426 classes.
- A validação cruzada usa `src.validacao.validacao_agrupada`, agrupada por gênero.

## Como comparar os classificadores (Etapa 2)

Requer as sementes da Etapa 1 em `data/seeds/`. Rode:

```bash
python -m src.experimento_etapa2   # cerca de 4 minutos (8 núcleos)
```

Compara k-NN, SVM RBF e Random Forest (grades em `src/modelos.py`, cada um com
e sem `StandardScaler` dentro de um `Pipeline`):

- treina **só** com `semente_reduzida.csv`; o script para se a semente tiver
  algum índice do teste, dos gêneros bloqueados ou do oculto;
- validação agrupada por gênero (`validacao_agrupada`), 5 dobras, repetida com
  as sementes 42..51, com **as mesmas dobras para todas as configurações**;
- pontuação única: maior = mais provável morcego (`src.modelos.pontuar`);
- o grupo `oculto` é só diagnóstico (recall por família/gênero, falsos
  positivos, vizinhos do k-NN) e **não** escolhe hiperparâmetro nem modelo.

Saídas em `results/`:

| Arquivo | Conteúdo |
|---|---|
| `etapa2_comparacao.csv` | todas as configurações: AP, F1 máximo, recall com precisão ≥ 0,95, sensibilidade/precisão/acurácia no limiar, limiares (média ± desvio entre repetições) |
| `etapa2_oculto_por_familia.csv` | recall no oculto por família e por gênero (`nivel`), para o melhor de cada modelo |
| `etapa2_oculto_vizinhos.csv` | 3 vizinhos mais próximos na semente de cada morcego do oculto (k-NN) |
| `etapa2_importancias_rf.csv` | importância das características da melhor Random Forest |
| `relatorio_etapa2.md` | tabela comparativa, limiares sugeridos, regra de desempate sugerida, diagnóstico do oculto e limites |
| `figuras/etapa2_*.png` | curvas precisão-recall, boxplot da AP, recall por família no oculto |

O relatório sugere o modelo e o limiar, mas a escolha final é da equipe.

## Enriquecimento iterativo (Etapa 3)

O classificador está fixo em `src/config_modelo.py` (decisão da equipe a partir
de `results/relatorio_etapa2.md`): k-NN `knn_frac`, k=3, votos por distância,
manhattan, sem escala, limiar 0.6728.

```bash
python -m src.enriquecimento --auto --ate-convergir   # simulação (~45 s), em data/rounds/auto/
python -m src.enriquecimento --rodada 1               # rodada manual, em data/rounds/rodada_01/
python -m src.enriquecimento --rodada 2               # incorpora as decisões da rodada 1
```

| Opção | Padrão | Efeito |
|---|---|---|
| `--limiar` | 0.6728 | pontuação mínima dos candidatos de alta confiança |
| `--max-incertos` | 30 | máximo de incertos (0 < pontuação < limiar, mais próximos do limiar) |
| `--max-revisao` | 100 | máximo de candidatos sem rótulo em `revisao_manual.csv` |
| `--max-rodadas` | 10 | limite do modo automático |
| `--sobrescrever` | — | permite refazer uma rodada com decisões preenchidas ou com rodadas posteriores |

Cada rodada treina na base atual (rodada 0 = `semente_reduzida.csv`), pontua
todas as instâncias fora da base e fora dos gêneros bloqueados e verifica os
candidatos:

- **do grupo `oculto`:** verificação automática pelo gabarito; entram na base
  (morcego confirmado ou negativo difícil);
- **sem rótulo:** vão para `revisao_manual.csv`, com os 3 vizinhos mais
  próximos na base. Preencha `decisao_humana` com `1` (morcego), `0` (não é
  morcego) ou `duvida` (fica fora), e use `observacao` para a fonte consultada.
  A decisão só entra na base quando a rodada seguinte é executada; linhas
  vazias continuam pendentes. Valor inválido para a rodada com erro.

A base **só cresce com rótulos verificados** (`origem` = `gabarito` ou
`humano`). Instâncias do teste e dos gêneros bloqueados são conferidas antes e
depois de cada rodada. Converge quando uma rodada não confirma positivo novo e
não há pendências (no modo automático as pendências sem rótulo não podem ser
resolvidas e ficam só registradas).

Arquivos em `data/rounds/rodada_NN/` (ou `data/rounds/auto/rodada_NN/`):
`base.csv` (`indice, rotulo, origem, rodada`), `candidatos.csv`,
`revisao_manual.csv` e `metricas.json`. Saídas em `results/`:
`etapa3_historico.csv`, `relatorio_etapa3.md` e as figuras
`etapa3_positivos_por_rodada.png`, `etapa3_recall_familia_por_rodada.png` e
`etapa3_umap.png`.

## Modelo final e avaliação (Etapa 4)

O modelo final é o classificador de `src/config_modelo.py`, treinado com a base
da última rodada manual **convergida** (`data/rounds/rodada_NN/base.csv`). O
limiar (0.6728) e os hiperparâmetros são fixos; nada é ajustado olhando o teste.

```bash
python -m src.avaliacao_final                                   # só preparação (não lê o teste)
python -m src.avaliacao_final --rodada-final 3 --avaliar-teste  # avaliação completa
```

Sem `--avaliar-teste`, o script valida a rodada, confere vazamento contra os
gêneros bloqueados, treina o modelo e mostra um resumo; nada é gravado em
`results/` e `teste_verificado.csv` não é aberto.

Com `--avaliar-teste`:

- confere o SHA256 do teste contra `teste_verificado.sha256` (para com erro se
  não bater) e registra o uso em `results/etapa4_log_uso_teste.csv` (data e
  hora, hash, configuração, limiar, rodada). Se já houver uso anterior, avisa:
  **repetir a avaliação enquanto se altera o modelo invalida o teste**;
- avalia no teste congelado: matriz de confusão; sensibilidade,
  especificidade, precisão e acurácia com IC 95% de Wilson; F1 e AP com
  bootstrap estratificado (2000 reamostragens, `--n-bootstrap`); linha de base
  "tudo 0"; negativos por origem e falsos positivos por ordem;
- no conjunto supervisionado final (base ∪ teste): validação agrupada por
  gênero (5 dobras × sementes 42..51) e *leave-one-genus-out* dos gêneros de
  morcego;
- análise de erros com os 3 vizinhos mais próximos e pontuações das 8426
  instâncias para a Etapa 5.

Saídas em `results/`: `etapa4_metricas_teste.csv`,
`etapa4_matriz_confusao_teste.csv`, `etapa4_cv_conjunto_final.csv`,
`etapa4_logo_generos.csv`, `etapa4_erros.csv`, `etapa4_pontuacoes_todas.csv`,
`etapa4_log_uso_teste.csv`, `relatorio_etapa4.md` e as figuras `etapa4_*.png`.

## Estrutura de pastas

```
data/
  raw/        Names.txt e wpc50.dat originais (somente leitura)
  external/   listas taxonômicas fornecidas pela equipe (Etapa 1)
  seeds/      sementes rotuladas e teste verificado (Etapa 1)
  rounds/     rótulos de cada rodada de enriquecimento (Etapa 3)
docs/         plano de execução, enunciado e fontes taxonômicas
notebooks/    notebooks de apoio
relatorio/    relatório final
results/      tabelas e relatórios gerados
  figuras/    figuras geradas
src/
  carregar_dados.py   carga e validação dos dados
  exploracao.py       exploração da Etapa 0
  supervisao.py       supervisão por gênero: gabarito, semente reduzida, teste e bloqueio
  validacao.py        validação cruzada agrupada por gênero e métricas de prospecção
  modelos.py          pipelines e grades de k-NN, SVM RBF e Random Forest (Etapa 2)
  experimento_etapa2.py  comparação dos classificadores e diagnóstico no oculto (Etapa 2)
  config_modelo.py    classificador escolhido e limiar padrão (Etapa 3+)
  enriquecimento.py   rodadas de enriquecimento, revisão manual e simulação (Etapa 3)
  avaliacao_final.py  modelo final, avaliação no teste congelado e análise de erros (Etapa 4)
tests/        testes (pytest)
  fixtures/   taxonomia fictícia, só para testes
```

## Convenção de índice

- Os dois arquivos estão alinhados pela posição: a linha *i* de `Names.txt`
  corresponde à linha *i* de `wpc50.dat`. **Nunca reordene os registros.**
- O índice do projeto é o número da linha original na **base 0**, definido pela
  constante `BASE_INDICE` em `src/carregar_dados.py`. A primeira linha do arquivo
  tem `indice = 0`.
- `carregar_tudo()` também devolve `indice_base1` (= `indice + 1`) para
  entregas que pedirem o "número de ordem" a partir de 1.
- **Toda saída final deve informar qual base usa** (no cabeçalho do CSV ou no
  nome da coluna).

## Regras de equipe

- Um branch por pessoa; nada vai direto para `main`.
- Toda mudança entra por *pull request* revisado por pelo menos uma outra pessoa.
- `random_state=42` em tudo que for aleatório.
- `data/raw/` nunca é alterado. Dados derivados vão para `data/seeds/`,
  `data/rounds/` ou `results/`.
