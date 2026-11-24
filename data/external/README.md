# Fontes taxonômicas externas

Os arquivos desta pasta são fornecidos pela equipe a partir de fontes
taxonômicas publicadas. **Não preencha de memória**: cada linha precisa vir de
uma fonte citável, registrada em `fonte` e `data_acesso` e em
[`docs/fontes_taxonomicas.md`](../../docs/fontes_taxonomicas.md).

## `generos_mamiferos.csv` (obrigatório)

Uma linha por gênero de mamífero, com **todos** os gêneros da classe Mammalia
(não só os que aparecem no `Names.txt`).

| coluna | conteúdo | exemplo |
|---|---|---|
| `genero` | nome do gênero, com inicial maiúscula, sem espaços | `Myotis` |
| `ordem` | ordem taxonômica; morcegos são **exatamente** `Chiroptera` | `Chiroptera` |
| `familia` | família taxonômica | `Vespertilionidae` |
| `fonte` | nome curto e versão da fonte | `MDD v2.0` |
| `data_acesso` | data em que os dados foram baixados, formato `AAAA-MM-DD` | `2026-11-23` |

- Gênero repetido com ordem ou família diferente gera aviso; vale a primeira linha.
- Espaços nas células são removidos na leitura.
- O código rotula: `ordem == "Chiroptera"` → 1; qualquer outra ordem → 0;
  gênero fora da lista → sem rótulo.

## `sinonimos_generos.csv` (opcional)

Gêneros antigos que aparecem no `Names.txt` e não casam com a lista atual.

| coluna | conteúdo |
|---|---|
| `genero_antigo` | gênero como está no `Names.txt` |
| `genero_atual` | gênero válido, presente em `generos_mamiferos.csv` |

O gênero antigo é trocado pelo atual antes do cruzamento, e o atual também é
usado como grupo na validação (espécies sinônimas não podem ficar em treino e
teste ao mesmo tempo).

## Onde obter

- **Mammal Diversity Database (MDD)** — https://www.mammaldiversity.org/ —
  planilha completa com ordem, família e gênero de todas as espécies; a fonte
  mais direta para este arquivo.
- **NCBI Taxonomy** — https://www.ncbi.nlm.nih.gov/taxonomy — útil para
  conferir sinônimos e nomes do GenBank, de onde vêm os genomas mitocondriais.
- **GBIF Backbone** / **ITIS** — https://www.gbif.org/ e https://www.itis.gov/ —
  conferência de sinônimos e casos duvidosos.

Ao reduzir a planilha de espécies a gêneros, mantenha uma linha por gênero e
registre a versão exata da fonte (ex.: `MDD v2.0`) e a data de download.
