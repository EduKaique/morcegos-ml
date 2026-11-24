# Fontes taxonômicas

Registro das fontes usadas na supervisão por gênero (Etapa 1). Preencher antes
de gerar `data/seeds/` e atualizar a cada mudança em `data/external/`.

## Fontes consultadas

| Fonte | Versão | URL | Data de acesso | Quem baixou | Quem conferiu | Arquivo gerado |
|---|---|---|---|---|---|---|
| Mammal Diversity Database | v2.5 | https://www.mammaldiversity.org/ | 2026-11-23 | Eduardo Kaique Alberico | | `data/external/generos_mamiferos.csv` |
| | | | AAAA-MM-DD | | | `data/external/sinonimos_generos.csv` |

## Critérios de inclusão

- Quais níveis foram incluídos (ex.: todos os gêneros válidos da classe Mammalia na versão citada):
- Como a planilha de espécies foi reduzida a uma linha por gênero:
- Como foram tratados gêneros com ordem/família divergente entre fontes:
- Gêneros extintos incluídos? (sim/não, e por quê):

## Sinônimos de gênero

| Gênero no Names.txt | Gênero atual | Fonte da sinonímia | Quem decidiu | Data |
|---|---|---|---|---|
| | | | | |

## Decisões sobre casos ambíguos

Casos listados em `data/seeds/casos_ambiguos.csv` (sp, híbridos, `nomen_nudum`,
subespécies e os 4 nomes de gênero suspeito, linhas 8422 a 8425). Registrar
aqui as decisões que mudarem o tratamento automático.

| Índice (base 0) | Nome | Decisão | Justificativa | Quem decidiu | Data |
|---|---|---|---|---|---|
| | | | | | |

## Modo semente reduzida (gabarito)

A taxonomia do MDD v2.5 é usada como **gabarito** (`data/seeds/gabarito.csv`):
serve para verificar candidatos e medir métricas, mas só o grupo `semente` entra
no treino da rodada 0. Parâmetros da geração atual (`python -m src.supervisao`):

| Parâmetro | Valor | Significado |
|---|---|---|
| `random_state` | 42 | teste: `default_rng(42)`; semente/oculto: `default_rng((42, 1))` |
| `--fracao-semente` | 0.35 | fração dos gêneros fora do teste que vão para a semente (morcegos por família, outros mamíferos por ordem, com ao menos 1 gênero de Eulipotyphla) |
| `--max-especies-teste` | 9 | gêneros com 10+ instâncias no Names.txt (ex.: *Myotis*) nunca vão para o teste |
| `--min-positivos-teste` | 5 | após 1 gênero por família com 2+ gêneros, sorteia mais gêneros de morcego até 5 positivos |
| `fracao` (teste de negativos) | 0.2 | como antes |
| `--n-negativos` | 300 | negativos prováveis (sem gabarito) |

## Histórico do teste verificado

O teste é congelado por SHA256 (`data/seeds/teste_verificado.sha256`). Só a
Pessoa A pode regerá-lo (`--sobrescrever-teste`), com aviso ao grupo.

| Data | SHA256 (8 primeiros) | Motivo | Quem |
|---|---|---|---|
| 2026-11-23 | 68bc13a7 | 1ª geração com o MDD v2.5, modo semente reduzida (189 instâncias: 5 morcegos de 5 gêneros, 184 negativos; 135 gêneros bloqueados) | |
