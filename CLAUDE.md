# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Projeto acadêmico (DS803, TADS/UFPR): classificar as 8426 instâncias de `data/raw/wpc50.dat` (50 características de genomas mitocondriais) em `1 = morcego` / `0 = outro`, usando os nomes científicos de `data/raw/Names.txt` como supervisão, com enriquecimento iterativo (semi-supervisionado). Código, nomes, mensagens e documentação são em **português** — mantenha esse padrão. O plano por etapas (0 a 8) está em `docs/plano_execucao_morcegos.md`; consulte-o antes de iniciar uma nova etapa.

## Comandos

Python 3.12, venv em `.venv` (Windows: `.venv\Scripts\Activate.ps1`). Sempre rode da raiz do repositório, como módulo (`python -m src.<módulo>`), pois os caminhos derivam de `DIR_RAIZ` em `src/carregar_dados.py` e o `pytest.ini` usa `pythonpath = .`.

```bash
pip install -r requirements.txt
python -m src.exploracao                     # Etapa 0: estatísticas, nomes fora do padrão, figuras
python -m src.supervisao                     # Etapa 1: sementes + teste verificado (padrão 300 negativos prováveis)
python -m src.supervisao --n-negativos 500
python -m src.supervisao --fracao-semente 0.35 --max-especies-teste 9 --min-positivos-teste 5
python scripts/montar_generos_mamiferos.py --mdd data/external/MDD_v2.5.csv --fonte "..." --data-acesso AAAA-MM-DD --names data/raw/Names.txt
pytest                                       # todos os testes
pytest tests/test_supervisao.py::nome_do_teste   # um teste
```

Não há linter configurado.

## Arquitetura

Pipeline em módulos de `src/`, cada etapa consumindo a saída da anterior:

1. **`carregar_dados.py`** — `carregar_tudo()` lê e valida os dois arquivos brutos (8426 linhas, 50 colunas) e devolve o DataFrame com `indice`, `indice_base1`, `nome`, `genero` etc. `extrair_genero`/`listar_nomes_fora_do_padrao` classificam nomes problemáticos (sp., híbridos, nomen nudum, minúsculas), que a Etapa 1 usa para separar casos ambíguos.
2. **`supervisao.py`** — cruza o gênero de cada nome com `data/external/generos_mamiferos.csv` (após aplicar `sinonimos_generos.csv`): `ordem == "Chiroptera"` → 1, outra ordem de mamífero → 0, gênero fora da lista → sem rótulo (uma amostra vira `negativo_provavel`, **não verificado**). Separa um teste verificado por **gêneros inteiros** e divide o resto em `semente` (treino da rodada 0) e `oculto` (só gabarito, para medir o enriquecimento); grava tudo em `data/seeds/` (`gabarito.csv`, `semente_reduzida.csv` = `treino_inicial.csv`, `generos_bloqueados.csv`, ...) + `results/relatorio_supervisao.md`. `instancias_bloqueadas(df)` dá todas as instâncias dos gêneros do teste. `executar(...)` aceita caminhos, o que os testes usam para rodar em diretórios temporários.
3. **`validacao.py`** — `validacao_agrupada()` gera dobras `StratifiedGroupKFold` agrupadas por `genero_atual`. Toda comparação de modelos (Etapa 2+) deve usar esta função. `avaliar_candidatos()` mede precisão/recall acumulados de candidatos no pool oculto.
4. `scripts/montar_generos_mamiferos.py` — reduz o checklist do MDD (uma linha por espécie; `MDD_v2.5.csv` é ignorado pelo git) a uma linha por gênero, normalizando ordem/família para capitalização (`CHIROPTERA` → `Chiroptera`).

Etapas futuras (classificadores, enriquecimento em `data/rounds/`, modelo final, `classes_finais.csv`/`indices_positivos.txt`) ainda não estão implementadas.

### Invariantes que não podem ser quebradas

- **Alinhamento por posição:** a linha *i* de `Names.txt` corresponde à linha *i* de `wpc50.dat`. Nunca reordene nem filtre sem preservar `indice`. O índice é **base 0** (`BASE_INDICE`); toda saída final deve declarar qual base usa.
- **`data/raw/` é somente leitura.** Derivados vão para `data/seeds/`, `data/rounds/` ou `results/`.
- **Vazamento por gênero:** espécies do mesmo gênero são quase duplicatas. Nenhum gênero pode estar em treino e validação/teste ao mesmo tempo. Todas as instâncias dos gêneros em `teste_verificado.csv` (coluna `genero_atual`) ficam fora do treino **e do enriquecimento**, mesmo as que não estão no arquivo.
- **Teste congelado:** `teste_verificado.sha256` congela o teste; `supervisao` aborta se uma nova execução gerar conteúdo diferente. Só use `--sobrescrever-teste` com aval da equipe.
- **Taxonomia citável:** linhas de `data/external/*.csv` devem vir de fonte publicada, com `fonte`/`data_acesso` preenchidos e registradas em `docs/fontes_taxonomicas.md` — nunca preencha gêneros de memória.
- `random_state=42` em tudo que for aleatório (`RANDOM_STATE` em cada módulo).

### Testes

`tests/conftest.py` roda a Etapa 1 inteira uma vez por sessão em `tmp_path` (nunca em `data/seeds/`). Usa a lista real `data/external/generos_mamiferos.csv` se ela tiver linhas; caso contrário, cai no fixture fictício `tests/fixtures/` (que não é taxonomia verificada). Testes portanto podem mudar de comportamento conforme a lista real.

## Fluxo de equipe

Um branch por pessoa, nada direto em `main`; toda mudança entra por PR revisado por outra pessoa. Figuras em `results/figuras/*.png` não são versionadas (são regeneradas).
