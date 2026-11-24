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
python -m src.experimento_etapa2             # Etapa 2: k-NN x SVM RBF x RF (~4 min)
python -m src.enriquecimento --auto --ate-convergir   # Etapa 3: simulação só com gabarito (data/rounds/auto/)
python -m src.enriquecimento --rodada 1      # Etapa 3: rodada manual (gera revisao_manual.csv)
python -m src.avaliacao_final                # Etapa 4: só preparação (não lê o teste)
python -m src.avaliacao_final --rodada-final 3 --avaliar-teste   # Etapa 4: avaliação no teste (registra uso)
python scripts/gerar_classes_finais.py --rodada-final 3 --base-lista 1   # Etapa 5: classes_finais.csv + indices_positivos.txt
python -m src.relatorio_etapa5 --rodada-final 3   # Etapa 5: verificações, relatório e figuras (não avalia o teste)
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

5. **`modelos.py`** — `gerar_configuracoes(grade)` monta as configurações (k-NN por fração ou razão de distâncias `RazaoDistanciaKNN`, SVM RBF, RF), cada uma sem e com `StandardScaler`, sempre num `Pipeline` (`construir_pipeline`). `pontuar(pipe, X)` é a interface única: `decision_function` se existir, senão `predict_proba[:, 1]`.
6. **`experimento_etapa2.py`** — Etapa 2: `verificar_sem_vazamento` antes de ajustar; `gerar_dobras` (sementes 42..51) uma vez para todas as configurações; métricas fora da amostra por repetição; melhor de cada modelo por AP; diagnóstico no oculto; grava `results/etapa2_*.csv`, `relatorio_etapa2.md` e figuras. `executar(dir_seeds, dir_resultados, seeds, grade)` aceita caminhos e grade reduzida (usado nos testes).

7. **`config_modelo.py`** — classificador fixo escolhido pela equipe na Etapa 2 (k-NN `knn_frac`, k=3, distância, manhattan, sem escala; `LIMIAR_PADRAO = 0.6728`). Use `construir_modelo()` em vez de repetir os parâmetros.
8. **`enriquecimento.py`** — Etapa 3: `executar_rodada(n, dir_modo, ctx, modo)` lê a base da rodada anterior (+ decisões humanas de `revisao_manual.csv`), treina, seleciona candidatos (alta confiança ≥ limiar; incertos com 0 < pontuação < limiar), verifica pelo gabarito (só grupo `oculto`) e grava `data/rounds/rodada_NN/` (`base.csv`, `candidatos.csv`, `revisao_manual.csv`, `metricas.json`). `executar_auto` simula em `data/rounds/auto/` (recomeça do zero). `gerar_saidas` regera `results/etapa3_historico.csv`, `relatorio_etapa3.md` e figuras a partir de todas as rodadas gravadas.

9. **`avaliacao_final.py`** — Etapa 4: `carregar_base_final` (rodada manual convergida) → `verificar_base` (vazamento) → modelo final. Com `avaliar_teste=True`: `ler_teste_verificado` (única porta de leitura do teste, confere SHA256), `registrar_uso_teste`, métricas com `wilson` (proporções) e `bootstrap_f1_ap`, `cv_conjunto_final` e `logo_generos` no conjunto base ∪ teste, `tabela_erros`, `pontuacoes_todas` (8426 linhas para a Etapa 5); grava `results/etapa4_*` e `relatorio_etapa4.md`.

10. **`scripts/gerar_classes_finais.py`** — Etapa 5: classe de cada uma das 8426 instâncias por prioridade gabarito → teste verificado → base final → modelo final (pontuação ≥ limiar); grava `results/classes_finais.csv` e `results/indices_positivos.txt`.
11. **`relatorio_etapa5.py`** — Etapa 5: `verificar_consistencia` (para com `ErroConsistencia`), `montar_morcegos` (`origem_entrada` = `semente`, `rodada_N` ou `teste_bloqueado`), `analisar_sem_rotulo` (desempate só de ordenação pela razão de distâncias), `segunda_opiniao` (melhor SVM e RF da Etapa 2, não altera rótulo), `carregar_umap_cache` (nunca calcula UMAP); grava `results/etapa5_morcegos.csv`, `relatorio_etapa5.md` e `figuras/etapa5_*.png`.

### Convenções da modelagem (Etapa 2+)

- Pontuação: **maior = mais provável morcego**, para todo modelo (use `src.modelos.pontuar`).
- Toda comparação usa as **mesmas dobras** para todos os modelos (`gerar_dobras`); repetições com as sementes 42..51.
- O grupo `oculto` é só diagnóstico: nunca escolhe hiperparâmetro, modelo ou limiar.
- Na Etapa 2 o treino é só `semente_reduzida.csv`.
- Etapa 3: a base só cresce com origem verificada (`gabarito` ou `humano`), nunca com a predição sozinha. `revisao_manual.csv` aceita `decisao_humana` = `1`, `0`, `duvida` ou vazio (pendente); `duvida` sai do pool nas rodadas seguintes. Não refaça uma rodada com decisões preenchidas (o script recusa sem `--sobrescrever`).
- As rodadas manuais (`data/rounds/rodada_NN/`) e a simulação (`data/rounds/auto/`) ficam separadas; o `--auto` apaga e refaz `data/rounds/auto/`.
- Etapa 4: o teste congelado só é lido com `--avaliar-teste`, sempre por `ler_teste_verificado` (SHA256 conferido); cada uso vira uma linha em `results/etapa4_log_uso_teste.csv`. Nunca ajuste limiar, k ou qualquer hiperparâmetro olhando o teste — repetir a avaliação enquanto se altera o modelo invalida o teste. O modelo final exige uma rodada manual convergida.
- Etapa 5: `indices_positivos.txt` usa **base 1** (número de ordem pedido no enunciado; `indice_base1 = indice_base0 + 1`). A Etapa 5 **não reavalia o teste**: sem métricas no teste e sem gravar em `etapa4_log_uso_teste.csv` (ler os rótulos por `ler_teste_verificado` para compor as classes é permitido). Nenhum rótulo muda no relatório. Não calcule UMAP novo (MemoryError na máquina da equipe): use PCA ou o cache conferido.

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
