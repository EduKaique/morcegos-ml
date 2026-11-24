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
