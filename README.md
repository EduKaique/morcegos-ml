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
pytest                     # testes de carga e alinhamento
```

A exploração grava `results/estatisticas_descritivas.csv`,
`results/nomes_fora_do_padrao.csv` e as figuras `pca_2d.png`, `umap_2d.png` e
`hist_especies_por_genero.png` em `results/figuras/`. As figuras não são
versionadas, porque o script as gera de novo.

## Estrutura de pastas

```
data/
  raw/        Names.txt e wpc50.dat originais (somente leitura)
  seeds/      sementes rotuladas (Etapa 1)
  rounds/     rótulos de cada rodada de enriquecimento (Etapa 3)
docs/         plano de execução e enunciado
notebooks/    notebooks de apoio
relatorio/    relatório final
results/      tabelas geradas
  figuras/    figuras geradas
src/
  carregar_dados.py   carga e validação dos dados
  exploracao.py       exploração da Etapa 0
tests/        testes (pytest)
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
