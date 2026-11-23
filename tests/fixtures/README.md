# Fixtures de teste

**Somente para os testes automatizados. Não é taxonomia verificada.**

- `generos_mamiferos_teste.csv`: poucos gêneros que existem no `Names.txt`, com
  famílias e várias ordens **fictícias** (`Familia_teste_*`, `Ordem_teste_*`).
  Serve para exercitar o código: famílias de morcego com 2+ gêneros e com um só,
  ordens de mamíferos diferentes (inclui `Eulipotyphla`), gêneros com nomes
  ambíguos (*Rhinolophus*, *Uropsilus*, *Mus*), espaços nas células e um gênero
  que não existe no `Names.txt`.
- `sinonimos_generos_teste.csv`: um sinônimo **fictício** (`Hylomys` →
  `Erinaceus`) para testar o mapeamento.

Os testes usam `data/external/generos_mamiferos.csv` quando ele tiver linhas, e
estes fixtures quando não tiver. Nunca use estes arquivos para gerar `data/seeds/`.
