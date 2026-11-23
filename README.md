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
pytest                     # testes (usam um fixture enquanto a lista real não existir)
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
   python -m src.supervisao                  # padrão: 300 negativos prováveis
   python -m src.supervisao --n-negativos 500
   ```

   Sem a lista (só o cabeçalho), o script para com uma mensagem e não grava nada.
3. Leia `results/relatorio_supervisao.md` e revise `data/seeds/casos_ambiguos.csv`.

Arquivos gerados em `data/seeds/`, na ordem em que são produzidos:

| Arquivo | Conteúdo |
|---|---|
| `positivos.csv` | todos os morcegos rotulados pelo gênero (`ordem == Chiroptera`) |
| `negativos.csv` | mamíferos não-morcegos (`mamifero_por_genero`) e a amostra de `negativo_provavel` |
| `treino_inicial.csv` | rotulados que podem ser usados no treino |
| `teste_verificado.csv` | gêneros inteiros reservados, sem nomes ambíguos |
| `casos_ambiguos.csv` | nomes ambíguos e suspeitos, a decisão tomada e uma coluna `observacao` para a equipe |
| `teste_verificado.sha256` | hash que congela o teste |

Colunas: `indice` (base 0), `nome`, `genero` (como está no `Names.txt`),
`genero_atual` (após sinônimos; é o grupo usado na validação), `ordem`,
`familia`, `rotulo` (1 morcego, 0 outro) e `origem`.

- **Teste congelado:** se o teste já existe e uma nova execução geraria outro
  conteúdo, o script para. Só com o aval da equipe use `--sobrescrever-teste`.
- **Negativos prováveis não são verificados:** são instâncias de gêneros fora da
  lista de mamíferos; um mamífero ausente da lista pode estar entre eles.
- **Rótulos do teste verificado na entrega final:** como foram verificados por
  taxonomia, esses rótulos também são usados diretamente no arquivo final das
  8426 classes. Para avaliar modelos, eles ficam fora do treino e do
  enriquecimento, e o mesmo vale para **todas as instâncias dos gêneros do
  teste** (coluna `genero_atual` de `teste_verificado.csv`), mesmo as que não
  estão no arquivo: rotulá-las no enriquecimento vazaria o gênero.
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
  supervisao.py       supervisão por gênero, sementes e teste verificado
  validacao.py        validação cruzada agrupada por gênero
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
