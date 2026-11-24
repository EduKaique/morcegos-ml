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

## Histórico do teste verificado

O teste é congelado por SHA256 (`data/seeds/teste_verificado.sha256`). Só a
Pessoa A pode regerá-lo (`--sobrescrever-teste`), com aviso ao grupo.

| Data | SHA256 (8 primeiros) | Motivo | Quem |
|---|---|---|---|
| | | | |
