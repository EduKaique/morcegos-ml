"""Supervisão inicial por gênero (Etapa 1), no modo semente reduzida.

Cruza o gênero de cada nome do ``Names.txt`` com uma lista externa de gêneros
de mamíferos (``data/external/generos_mamiferos.csv``). A taxonomia completa é
o **gabarito**: serve para verificar candidatos e medir métricas, mas só uma
parte dela (a **semente reduzida**) é usada no treino da rodada 0.

Cada gênero rotulado cai em um de três grupos:

- ``teste``: gêneros inteiros reservados ao teste verificado (congelado por
  SHA256) e bloqueados no treino e no enriquecimento;
- ``semente``: conhecidos na rodada 0;
- ``oculto``: têm gabarito, mas não entram no treino da rodada 0; é o pool em
  que o enriquecimento (Etapa 3) é avaliado.

Uso: ``python -m src.supervisao`` a partir da raiz do repositório.
"""

import argparse
import hashlib
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.carregar_dados import DIR_RAIZ, carregar_tudo, listar_nomes_fora_do_padrao

RANDOM_STATE = 42
ORDEM_MORCEGO = "Chiroptera"
ORDEM_GARANTIDA_SEMENTE = "Eulipotyphla"  # insetívoros: negativos difíceis

DIR_EXTERNAL = DIR_RAIZ / "data" / "external"
DIR_SEEDS = DIR_RAIZ / "data" / "seeds"
DIR_RESULTADOS = DIR_RAIZ / "results"
ARQ_TAXONOMIA = DIR_EXTERNAL / "generos_mamiferos.csv"
ARQ_SINONIMOS = DIR_EXTERNAL / "sinonimos_generos.csv"
ARQ_BLOQUEADOS = DIR_SEEDS / "generos_bloqueados.csv"

COLUNAS_TAXONOMIA = ["genero", "ordem", "familia", "fonte", "data_acesso"]
COLUNAS_SINONIMOS = ["genero_antigo", "genero_atual"]
COLUNAS_SAIDA = ["indice", "nome", "genero", "genero_atual", "ordem", "familia", "rotulo", "origem"]
COLUNAS_COM_GRUPO = COLUNAS_SAIDA + ["grupo"]
COLUNAS_GABARITO = ["indice", "nome", "genero", "genero_atual", "ordem", "familia",
                    "rotulo_gabarito", "grupo"]
COLUNAS_BLOQUEADOS = ["genero_atual", "origem", "ordem", "familia", "n_instancias_names", "n_no_teste"]

# Motivos de listar_nomes_fora_do_padrao que tornam um nome ambíguo.
MOTIVOS_AMBIGUOS = {"sp", "nomen_nudum", "hibrido", "mais de dois termos"}
MOTIVO_SUSPEITO = "primeiro termo minusculo"

ORIGEM_MORCEGO = "chiroptera_por_genero"
ORIGEM_MAMIFERO = "mamifero_por_genero"
ORIGEM_PROVAVEL = "negativo_provavel"

GRUPO_SEMENTE = "semente"
GRUPO_OCULTO = "oculto"
GRUPO_TESTE = "teste"
GRUPOS = (GRUPO_SEMENTE, GRUPO_OCULTO, GRUPO_TESTE)

MIN_POSITIVOS_AVISO = 5


class ListaTaxonomicaAusente(ValueError):
    """A lista de gêneros de mamíferos não existe ou não tem linhas."""


# --------------------------------------------------------------------------
# Leitura das fontes externas
# --------------------------------------------------------------------------

def _ler_csv_limpo(caminho, colunas):
    """Lê um CSV como texto, valida as colunas e remove espaços das células."""
    tabela = pd.read_csv(caminho, dtype=str, keep_default_na=False)
    tabela.columns = tabela.columns.str.strip()
    faltando = [c for c in colunas if c not in tabela.columns]
    if faltando:
        raise ValueError(f"{caminho}: faltam as colunas {faltando}; esperadas {colunas}.")
    return tabela[colunas].apply(lambda col: col.str.strip())


def carregar_taxonomia(caminho=ARQ_TAXONOMIA):
    """Lê a lista de gêneros de mamíferos (genero, ordem, familia, fonte, data_acesso).

    Valida as colunas, remove espaços e descarta linhas sem gênero. Se um gênero
    aparece mais de uma vez com ordem ou família diferente, emite um aviso e
    mantém a primeira ocorrência.
    """
    tax = _ler_csv_limpo(caminho, COLUNAS_TAXONOMIA)
    tax = tax[tax["genero"] != ""]
    divergentes = tax.groupby("genero")[["ordem", "familia"]].nunique().max(axis=1) > 1
    if divergentes.any():
        lista = ", ".join(divergentes[divergentes].index)
        warnings.warn(f"Gêneros duplicados com ordem/família divergente (mantida a 1ª linha): {lista}")
    return tax.drop_duplicates("genero", keep="first").reset_index(drop=True)


def carregar_sinonimos(caminho=ARQ_SINONIMOS):
    """Lê a tabela opcional de sinônimos e devolve ``{genero_antigo: genero_atual}``.

    Arquivo ausente ou sem linhas resulta em dicionário vazio.
    """
    if not Path(caminho).exists():
        return {}
    sin = _ler_csv_limpo(caminho, COLUNAS_SINONIMOS)
    sin = sin[(sin["genero_antigo"] != "") & (sin["genero_atual"] != "")]
    return dict(zip(sin["genero_antigo"], sin["genero_atual"]))


# --------------------------------------------------------------------------
# Anotação e montagem dos rotulados
# --------------------------------------------------------------------------

def _eh_ambiguo(motivo):
    return bool(set(motivo.split("; ")) & MOTIVOS_AMBIGUOS)


def anotar_taxonomia(df, tax, sinonimos=None):
    """Anota o DataFrame de ``carregar_tudo()`` com a taxonomia externa.

    Colunas adicionadas:
      - ``genero_atual``: gênero após aplicar os sinônimos (usado no cruzamento
        e como grupo na separação treino/teste);
      - ``ordem``, ``familia``: vindas da lista (vazias se o gênero não casar);
      - ``rotulo_fonte``: 1 se ordem == Chiroptera, 0 se outro mamífero,
        ``<NA>`` se o gênero não está na lista;
      - ``motivo``: motivos de ``listar_nomes_fora_do_padrao`` ("" se nenhum);
      - ``ambiguo``: nome com sp, nomen_nudum, híbrido ou subespécie;
      - ``genero_suspeito``: primeiro termo em minúscula (ex.: ``diploid_``).
        Esses nomes NUNCA recebem rótulo.
    """
    sinonimos = sinonimos or {}
    out = df.copy()
    out["genero_atual"] = out["genero"].map(lambda g: sinonimos.get(g, g))
    out = out.join(tax.set_index("genero")[["ordem", "familia"]], on="genero_atual")

    motivos = listar_nomes_fora_do_padrao(df).set_index("indice")["motivo"]
    out["motivo"] = [motivos.get(i, "") for i in out["indice"]]
    out["ambiguo"] = out["motivo"].map(_eh_ambiguo)
    out["genero_suspeito"] = out["motivo"].str.contains(MOTIVO_SUSPEITO, regex=False)

    rotulo = (out["ordem"] == ORDEM_MORCEGO).astype("Int64")
    rotulo[out["ordem"].isna() | out["genero_suspeito"]] = pd.NA
    out["rotulo_fonte"] = rotulo
    return out


def contar_especies(df):
    """Número de instâncias do ``Names.txt`` por ``genero_atual`` (todas, com ou sem rótulo)."""
    return df.groupby("genero_atual").size()


def montar_negativos_outros(df, n=300, seed=RANDOM_STATE):
    """Amostra ``n`` instâncias sem rótulo e sem gênero suspeito como negativos prováveis.

    Recebe o DataFrame de ``anotar_taxonomia``. As instâncias NÃO são
    verificadas: a lista cobre só mamíferos, e um mamífero ausente da lista
    (sinônimo, erro) pode cair aqui. Retorna as linhas com ``rotulo = 0`` e
    ``origem = "negativo_provavel"``.
    """
    candidatos = df[df["rotulo_fonte"].isna() & ~df["genero_suspeito"]]
    amostra = candidatos.sample(n=min(n, len(candidatos)), random_state=seed).copy()
    amostra["rotulo"] = 0
    amostra["origem"] = ORIGEM_PROVAVEL
    return amostra.sort_values("indice")


def montar_rotulados(df, negativos_outros):
    """Junta mamíferos rotulados pela lista e negativos prováveis num só DataFrame.

    Adiciona ``rotulo`` (0/1) e ``origem`` (chiroptera_por_genero,
    mamifero_por_genero ou negativo_provavel), em ordem de ``indice``.
    """
    mamiferos = df[df["rotulo_fonte"].notna()].copy()
    mamiferos["rotulo"] = mamiferos["rotulo_fonte"].astype(int)
    mamiferos["origem"] = np.where(mamiferos["rotulo"] == 1, ORIGEM_MORCEGO, ORIGEM_MAMIFERO)
    rotulados = pd.concat([mamiferos, negativos_outros])
    return rotulados.sort_values("indice").reset_index(drop=True)


# --------------------------------------------------------------------------
# Separação do teste verificado por gêneros inteiros
# --------------------------------------------------------------------------

def _sortear(rng, generos, k):
    """Sorteia ``k`` gêneros (sem reposição) de uma lista, de forma reprodutível."""
    generos = sorted(generos)
    k = min(k, len(generos))
    return {str(g) for g in rng.choice(generos, size=k, replace=False)} if k > 0 else set()


def _um_por_grupo(rng, elegiveis, coluna):
    """Para cada valor de ``coluna`` com 2+ gêneros, sorteia 1 gênero."""
    escolhidos = set()
    for _, grupo in elegiveis.groupby(coluna, sort=True):
        generos = grupo["genero_atual"].unique()
        if len(generos) >= 2:
            escolhidos |= _sortear(rng, generos, 1)
    return escolhidos


def _elegiveis_teste(df_rotulado, n_especies, max_especies_teste):
    """Linhas que podem ir ao teste: nome não ambíguo e gênero com até ``max_especies_teste`` espécies."""
    n = df_rotulado["genero_atual"].map(n_especies).fillna(0)
    return df_rotulado[~df_rotulado["ambiguo"] & (n <= max_especies_teste)]


def familias_elegiveis(df_rotulado, n_especies, max_especies_teste=9):
    """Famílias de morcego com 2+ gêneros rotulados e ao menos 1 gênero elegível ao teste."""
    morcegos = df_rotulado[df_rotulado["origem"] == ORIGEM_MORCEGO]
    n_generos = morcegos.groupby("familia")["genero_atual"].nunique()
    elegiveis = _elegiveis_teste(morcegos, n_especies, max_especies_teste)
    com_elegivel = set(elegiveis["familia"])
    return sorted(f for f in n_generos[n_generos >= 2].index if f in com_elegivel)


def dividir_teste_por_genero(df_rotulado, n_especies, fracao=0.2, max_especies_teste=9,
                             min_positivos_teste=5, seed=RANDOM_STATE):
    """Reserva gêneros inteiros para o teste verificado.

    Devolve ``(fora_do_teste, teste, excluidos, reservados)``.

    Regras (determinísticas para o mesmo ``seed``):
      - só são elegíveis gêneros com ao menos um nome não ambíguo e com até
        ``max_especies_teste`` instâncias no ``Names.txt`` (ex.: *Myotis*, com
        28, nunca vai para o teste);
      - morcegos: 1 gênero elegível por família com 2+ gêneros rotulados; depois,
        se o teste tiver menos de ``min_positivos_teste`` morcegos, sorteia mais
        gêneros elegíveis, sempre deixando ao menos 1 gênero de cada família
        fora do teste;
      - mamíferos não-morcegos: 1 gênero por ordem com 2+ gêneros elegíveis,
        completando até ``fracao`` dos gêneros;
      - negativos prováveis: ``fracao`` dos gêneros elegíveis;
      - nos gêneros reservados, os nomes não ambíguos vão para o teste e os
        ambíguos vão para ``excluidos`` (fora do teste e do treino).
    O grupo é ``genero_atual``.
    """
    rng = np.random.default_rng(seed)
    elegiveis = _elegiveis_teste(df_rotulado, n_especies, max_especies_teste)
    por_origem = {o: elegiveis[elegiveis["origem"] == o] for o in
                  (ORIGEM_MORCEGO, ORIGEM_MAMIFERO, ORIGEM_PROVAVEL)}

    # Morcegos: 1 por família com 2+ gêneros, depois completa até o mínimo.
    morcegos = df_rotulado[df_rotulado["origem"] == ORIGEM_MORCEGO]
    generos_fam = morcegos.groupby("familia")["genero_atual"].unique()
    eleg_m = por_origem[ORIGEM_MORCEGO]
    reservados = set()
    for familia in familias_elegiveis(df_rotulado, n_especies, max_especies_teste):
        reservados |= _sortear(rng, eleg_m.loc[eleg_m["familia"] == familia, "genero_atual"].unique(), 1)

    def _n_positivos():
        return int(eleg_m["genero_atual"].isin(reservados).sum())

    while _n_positivos() < min_positivos_teste:
        candidatos = [
            g for g, fam in eleg_m[["genero_atual", "familia"]].drop_duplicates().itertuples(index=False)
            if g not in reservados
            and len(set(generos_fam[fam]) - reservados) >= 2  # a família fica com 1+ gênero fora
        ]
        if not candidatos:
            break
        reservados |= _sortear(rng, candidatos, 1)

    mamiferos = por_origem[ORIGEM_MAMIFERO]
    escolhidos = _um_por_grupo(rng, mamiferos, "ordem")
    todos = set(mamiferos["genero_atual"])
    faltam = round(fracao * len(todos)) - len(escolhidos)
    reservados |= escolhidos | _sortear(rng, todos - escolhidos, faltam)

    provaveis = set(por_origem[ORIGEM_PROVAVEL]["genero_atual"])
    reservados |= _sortear(rng, provaveis, round(fracao * len(provaveis)))

    no_teste = df_rotulado["genero_atual"].isin(reservados)
    teste = df_rotulado[no_teste & ~df_rotulado["ambiguo"]]
    excluidos = df_rotulado[no_teste & df_rotulado["ambiguo"]]
    fora_do_teste = df_rotulado[~no_teste]
    return fora_do_teste, teste, excluidos, reservados


# --------------------------------------------------------------------------
# Semente reduzida × oculto
# --------------------------------------------------------------------------

def _alocar_estratificado(rng, generos_por_estrato, fracao, garantir=()):
    """Sorteia ``round(fracao × total)`` gêneros, repartidos entre os estratos.

    A cota de cada estrato é ``fracao × nº de gêneros``; cada estrato recebe a
    parte inteira e as vagas restantes vão para os maiores restos (empates
    resolvidos pelo ``rng``). Estratos em ``garantir`` recebem ao menos 1
    gênero. Assim, estratos com um único gênero também podem ser sorteados.
    """
    estratos = {e: sorted(set(g)) for e, g in sorted(generos_por_estrato.items()) if len(g)}
    total = sum(len(g) for g in estratos.values())
    k = round(fracao * total)
    cotas = {e: fracao * len(g) for e, g in estratos.items()}
    n = {e: math.floor(c) for e, c in cotas.items()}
    for e in garantir:
        if e in n:
            n[e] = max(n[e], 1)
    resto = k - sum(n.values())
    if resto > 0:
        ordem = list(estratos)
        ordem = [ordem[i] for i in rng.permutation(len(ordem))]  # desempate aleatório
        ordem.sort(key=lambda e: cotas[e] - math.floor(cotas[e]), reverse=True)  # sort estável
        for e in [e for e in ordem if n[e] < len(estratos[e])][:resto]:
            n[e] += 1
    escolhidos = set()
    for e, generos in estratos.items():
        escolhidos |= _sortear(rng, generos, n[e])
    return escolhidos


def dividir_semente_oculto(rotulados, reservados, fracao_semente=0.35, seed=RANDOM_STATE):
    """Atribui o ``grupo`` (semente, oculto ou teste) a cada linha de ``rotulados``.

    - gêneros em ``reservados`` → ``teste`` (inclusive nomes ambíguos);
    - morcegos fora do teste: ``fracao_semente`` dos gêneros vão para a
      semente, estratificados por família; o resto fica ``oculto``;
    - mamíferos não-morcegos fora do teste: idem, estratificados por ordem,
      com ao menos 1 gênero de Eulipotyphla na semente;
    - negativos prováveis fora do teste → ``semente`` (não têm gabarito).

    Usa um gerador separado do teste (``default_rng((seed, 1))``): mudar
    ``fracao_semente`` não altera o teste verificado.
    """
    rng = np.random.default_rng((seed, 1))
    fora = rotulados[~rotulados["genero_atual"].isin(reservados)]

    def _estratos(origem, coluna):
        sub = fora[fora["origem"] == origem]
        return {str(k): v for k, v in sub.groupby(coluna)["genero_atual"].unique().items()}

    semente = _alocar_estratificado(rng, _estratos(ORIGEM_MORCEGO, "familia"), fracao_semente)
    semente |= _alocar_estratificado(rng, _estratos(ORIGEM_MAMIFERO, "ordem"), fracao_semente,
                                     garantir=(ORDEM_GARANTIDA_SEMENTE,))

    out = rotulados.copy()
    out["grupo"] = np.select(
        [out["genero_atual"].isin(reservados),
         (out["origem"] == ORIGEM_PROVAVEL) | out["genero_atual"].isin(semente)],
        [GRUPO_TESTE, GRUPO_SEMENTE],
        default=GRUPO_OCULTO,
    )
    return out


def montar_gabarito(rotulados):
    """Instâncias com rótulo taxonômico (sem os negativos prováveis), com ``rotulo_gabarito`` e ``grupo``."""
    gab = rotulados[rotulados["origem"] != ORIGEM_PROVAVEL].copy()
    gab["rotulo_gabarito"] = gab["rotulo"]
    return gab


def montar_generos_bloqueados(rotulados, teste, reservados, n_especies):
    """Uma linha por gênero do teste, com origem, ordem, família e contagens."""
    base = (rotulados[rotulados["genero_atual"].isin(reservados)]
            .groupby("genero_atual")[["origem", "ordem", "familia"]].first())
    base["n_instancias_names"] = n_especies.reindex(base.index).fillna(0).astype(int)
    base["n_no_teste"] = teste.groupby("genero_atual").size().reindex(base.index).fillna(0).astype(int)
    return base.reset_index().sort_values(["origem", "genero_atual"])[COLUNAS_BLOQUEADOS]


def instancias_bloqueadas(df, arq_bloqueados=ARQ_BLOQUEADOS):
    """Índices de TODAS as instâncias de ``df`` cujos gêneros estão reservados ao teste.

    Inclui instâncias sem rótulo (ex.: espécies de um gênero de negativo
    provável que não entraram na amostra) e nomes ambíguos. A Etapa 3 deve
    excluí-las dos candidatos e do treino; na entrega final elas recebem o
    rótulo do gabarito.

    ``df`` é o DataFrame de ``carregar_tudo()`` (anotado ou não). Se não tiver
    ``genero_atual``, os sinônimos de ``data/external`` são aplicados a
    ``genero``. Retorna uma lista ordenada de ``indice`` (base 0).
    """
    bloqueados = set(pd.read_csv(arq_bloqueados, dtype=str, keep_default_na=False)["genero_atual"])
    if "genero_atual" in df.columns:
        generos = df["genero_atual"]
    else:
        sinonimos = carregar_sinonimos()
        generos = df["genero"].map(lambda g: sinonimos.get(g, g))
    return sorted(int(i) for i in df.loc[generos.isin(bloqueados), "indice"])


# --------------------------------------------------------------------------
# Casos ambíguos, gravação e congelamento
# --------------------------------------------------------------------------

def montar_casos_ambiguos(df, rotulados, excluidos, reservados):
    """Tabela de nomes ambíguos e suspeitos com a decisão tomada para cada um.

    A coluna ``observacao`` fica vazia para a equipe preencher.
    """
    casos = df[df["ambiguo"] | df["genero_suspeito"]].copy()
    grupo = rotulados.set_index("indice")["grupo"]
    casos["grupo"] = casos["indice"].map(grupo)
    provaveis = set(rotulados.loc[rotulados["origem"] == ORIGEM_PROVAVEL, "indice"])
    casos["decisao"] = np.select(
        [casos["genero_suspeito"],
         casos["indice"].isin(excluidos["indice"]),
         casos["genero_atual"].isin(reservados),
         casos["indice"].isin(provaveis),
         casos["rotulo_fonte"].notna() & (casos["grupo"] == GRUPO_SEMENTE),
         casos["rotulo_fonte"].notna() & (casos["grupo"] == GRUPO_OCULTO)],
        ["sem rotulo (genero suspeito)",
         "excluido (genero reservado ao teste)",
         "sem rotulo; genero bloqueado (teste)",
         "negativo provavel; grupo semente",
         "rotulado por genero; grupo semente",
         "rotulado por genero; grupo oculto (so gabarito)"],
        default="sem rotulo (genero fora da lista)",
    )
    casos["observacao"] = ""
    colunas = ["indice", "nome", "genero", "genero_atual", "motivo", "rotulo_fonte", "decisao", "observacao"]
    return casos[colunas]


def _csv_bytes(tabela, colunas=COLUNAS_SAIDA, ordenar="indice"):
    """Serializa a tabela em CSV (UTF-8, quebra de linha LF) ordenada por ``ordenar``."""
    if ordenar:
        tabela = tabela.sort_values(ordenar)
    return tabela[colunas].to_csv(index=False, lineterminator="\n").encode("utf-8")


def sha256_bytes(conteudo):
    return hashlib.sha256(conteudo).hexdigest()


def verificar_congelamento(dir_seeds, conteudo_teste, sobrescrever_teste=False):
    """Impede sobrescrever um teste verificado congelado com conteúdo diferente.

    Se ``teste_verificado.sha256`` existe e o novo teste tem outro hash, levanta
    ``RuntimeError``, a menos que ``sobrescrever_teste=True``.
    """
    arq_hash = Path(dir_seeds) / "teste_verificado.sha256"
    if not arq_hash.exists() or sobrescrever_teste:
        return
    congelado = arq_hash.read_text(encoding="utf-8").split()[0]
    if congelado != sha256_bytes(conteudo_teste):
        raise RuntimeError(
            "O teste verificado já está congelado e o novo conteúdo é diferente. "
            "Use --sobrescrever-teste só com o aval da equipe."
        )


def salvar_sementes(dir_seeds, rotulados, semente, gabarito, bloqueados, conteudo_teste, casos):
    """Grava todos os arquivos de ``data/seeds/`` e o SHA256 do teste."""
    dir_seeds = Path(dir_seeds)
    dir_seeds.mkdir(parents=True, exist_ok=True)
    escrever = {
        "gabarito.csv": _csv_bytes(gabarito, COLUNAS_GABARITO),
        "semente_reduzida.csv": _csv_bytes(semente),
        "treino_inicial.csv": _csv_bytes(semente),
        "positivos.csv": _csv_bytes(rotulados[rotulados["rotulo"] == 1], COLUNAS_COM_GRUPO),
        "negativos.csv": _csv_bytes(rotulados[rotulados["rotulo"] == 0], COLUNAS_COM_GRUPO),
        "teste_verificado.csv": conteudo_teste,
        "generos_bloqueados.csv": _csv_bytes(bloqueados, COLUNAS_BLOQUEADOS, ordenar=None),
        "casos_ambiguos.csv": _csv_bytes(casos, list(casos.columns)),
    }
    for nome, conteudo in escrever.items():
        (dir_seeds / nome).write_bytes(conteudo)
    (dir_seeds / "teste_verificado.sha256").write_text(
        f"{sha256_bytes(conteudo_teste)}  teste_verificado.csv\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Relatório
# --------------------------------------------------------------------------

def _tabela_md(tabela):
    """Converte um DataFrame (ou Series) em tabela Markdown simples."""
    if isinstance(tabela, pd.Series):
        tabela = tabela.reset_index()
    if tabela.empty:
        return "_(nenhum)_\n"
    linhas = ["| " + " | ".join(map(str, tabela.columns)) + " |",
              "|" + "---|" * len(tabela.columns)]
    linhas += ["| " + " | ".join(map(str, linha)) + " |" for linha in tabela.itertuples(index=False)]
    return "\n".join(linhas) + "\n"


def _contagem(tabela, coluna):
    """Instâncias e gêneros por valor de ``coluna``."""
    return (tabela.groupby(coluna, dropna=False)
            .agg(instancias=("indice", "size"), generos=("genero_atual", "nunique"))
            .reset_index())


def _pivo_grupos(tabela, coluna):
    """Tabela ``coluna`` × grupo com células "instâncias (gêneros)" e linha de total."""
    tabela = tabela.assign(**{coluna: tabela[coluna].fillna("-")})

    def _celula(sub):
        return f"{len(sub)} ({sub['genero_atual'].nunique()})" if len(sub) else "0"

    linhas = []
    for valor, sub in list(tabela.groupby(coluna, sort=True)) + [("**total**", tabela)]:
        linha = {coluna: valor}
        for g in GRUPOS:
            linha[g] = _celula(sub[sub["grupo"] == g])
        linha["total"] = _celula(sub)
        linhas.append(linha)
    return pd.DataFrame(linhas)


def generos_lista_sem_correspondencia(tax, df):
    """Gêneros da lista externa que não aparecem no Names.txt (após sinônimos)."""
    return tax[~tax["genero"].isin(set(df["genero_atual"]))][["genero", "ordem", "familia"]]


def calcular_avisos(semente, gabarito, teste, min_positivos_teste):
    """Avisos sobre grupos com poucos positivos ou sem Eulipotyphla na semente."""
    positivos = {
        GRUPO_SEMENTE: int((semente["rotulo"] == 1).sum()),
        GRUPO_OCULTO: int(((gabarito["grupo"] == GRUPO_OCULTO) & (gabarito["rotulo_gabarito"] == 1)).sum()),
        GRUPO_TESTE: int((teste["rotulo"] == 1).sum()),
    }
    avisos = [f"O grupo `{g}` tem só {n} positivo(s) (menos de {MIN_POSITIVOS_AVISO})."
              for g, n in positivos.items() if n < MIN_POSITIVOS_AVISO]
    if positivos[GRUPO_TESTE] < min_positivos_teste:
        avisos.append(f"Não foi possível completar o teste até {min_positivos_teste} positivos "
                      "sem quebrar as regras (limite de espécies ou família sem gênero fora do teste).")
    neg_semente = semente[semente["origem"] == ORIGEM_MAMIFERO]
    if ORDEM_GARANTIDA_SEMENTE not in set(neg_semente["ordem"]):
        avisos.append(f"Nenhum negativo de {ORDEM_GARANTIDA_SEMENTE} na semente.")
    return avisos


def gerar_relatorio(df, tax, rotulados, semente, gabarito, teste, excluidos, bloqueados, casos,
                    sem_corresp, params):
    """Monta o texto Markdown do relatório da supervisão."""
    pos = rotulados[rotulados["origem"] == ORIGEM_MORCEGO]
    mam = rotulados[rotulados["origem"] == ORIGEM_MAMIFERO]
    prov = rotulados[rotulados["origem"] == ORIGEM_PROVAVEL]
    negativos = rotulados[rotulados["rotulo"] == 0].copy()
    negativos["ordem"] = np.where(negativos["origem"] == ORIGEM_PROVAVEL, "(negativo provável)",
                                  negativos["ordem"])
    sem_rotulo = df["rotulo_fonte"].isna()
    casados = df[df["rotulo_fonte"].notna()]
    via_sinonimo = casados[casados["genero"] != casados["genero_atual"]]
    generos_sem = df.loc[sem_rotulo & ~df["genero_suspeito"], "genero_atual"].nunique()
    fontes = ", ".join(sorted(set(tax["fonte"]) - {""})) or "(não informada)"
    bloq_idx = df["genero_atual"].isin(set(bloqueados["genero_atual"]))
    bloq_sem_rotulo = int((bloq_idx & sem_rotulo).sum())
    avisos = calcular_avisos(semente, gabarito, teste, params["min_positivos_teste"])
    oculto = gabarito[gabarito["grupo"] == GRUPO_OCULTO]

    p = ["# Relatório da supervisão inicial (Etapa 1, modo semente reduzida)\n",
         f"Lista taxonômica: {len(tax)} gêneros; fonte(s): {fontes}. "
         f"Índices na base 0 (`BASE_INDICE`).\n",
         "Parâmetros: " + ", ".join(f"`{k}={v}`" for k, v in params.items()) + ".\n",
         "## Avisos\n",
         "\n".join(f"- {a}" for a in avisos) + "\n" if avisos else "_(nenhum)_\n",
         "## Grupos\n",
         "- `semente`: conhecido na rodada 0 (`semente_reduzida.csv` = `treino_inicial.csv`);\n"
         "- `oculto`: tem gabarito, mas não entra no treino da rodada 0; pool onde o "
         "enriquecimento é medido (`src.validacao.avaliar_candidatos`);\n"
         "- `teste`: gêneros inteiros reservados e bloqueados (`generos_bloqueados.csv`).\n",
         "Células: instâncias (gêneros).\n",
         "### Morcegos por família e grupo\n",
         f"Total: {len(pos)} instâncias de {pos['genero_atual'].nunique()} gêneros.\n",
         _tabela_md(_pivo_grupos(pos, "familia")),
         "### Negativos por ordem e grupo\n",
         f"Mamíferos não-morcegos: {len(mam)} instâncias de {mam['genero_atual'].nunique()} gêneros. "
         "A linha \"(negativo provável)\" é a amostra aleatória sem gabarito.\n",
         _tabela_md(_pivo_grupos(negativos, "ordem")),
         "### Resumo por grupo\n",
         _tabela_md(pd.DataFrame([
             {"grupo": GRUPO_SEMENTE, "positivos": int((semente["rotulo"] == 1).sum()),
              "negativos": int((semente["rotulo"] == 0).sum()), "generos": semente["genero_atual"].nunique()},
             {"grupo": GRUPO_OCULTO, "positivos": int((oculto["rotulo_gabarito"] == 1).sum()),
              "negativos": int((oculto["rotulo_gabarito"] == 0).sum()), "generos": oculto["genero_atual"].nunique()},
             {"grupo": GRUPO_TESTE, "positivos": int((teste["rotulo"] == 1).sum()),
              "negativos": int((teste["rotulo"] == 0).sum()), "generos": teste["genero_atual"].nunique()},
         ])),
         "## Teste verificado\n",
         f"{len(teste)} instâncias de {teste['genero_atual'].nunique()} gêneros inteiros "
         f"(positivos: {int((teste['rotulo'] == 1).sum())}, negativos: {int((teste['rotulo'] == 0).sum())}). "
         f"{len(excluidos)} nomes ambíguos de gêneros reservados ficaram fora do teste e do treino.\n",
         _tabela_md(_contagem(teste, ["origem", "ordem", "familia"]).fillna("-")),
         "Gêneros de morcego no teste: "
         + (", ".join(sorted(teste.loc[teste["rotulo"] == 1, "genero_atual"].unique())) or "_(nenhum)_") + ".\n",
         "## Gêneros bloqueados\n",
         f"{len(bloqueados)} gêneros (`data/seeds/generos_bloqueados.csv`), com {int(bloq_idx.sum())} "
         f"instâncias no Names.txt, das quais {bloq_sem_rotulo} sem rótulo. Todas ficam fora do "
         "treino e do enriquecimento (`src.supervisao.instancias_bloqueadas`).\n",
         _tabela_md(bloqueados.groupby("origem").agg(generos=("genero_atual", "size"),
                                                     instancias=("n_instancias_names", "sum")).reset_index()),
         "## Negativos prováveis\n",
         f"{len(prov)} instâncias (pedido: {params['n_negativos']}) de {prov['genero_atual'].nunique()} gêneros, "
         f"amostradas entre as instâncias sem rótulo (random_state={RANDOM_STATE}).\n",
         "> **Atenção:** os negativos prováveis NÃO são verificados. A lista cobre só "
         "mamíferos; um mamífero ausente da lista (sinônimo, erro de grafia) pode estar "
         "entre eles. Conferir os nomes antes de confiar nas métricas.\n",
         "## Instâncias sem rótulo\n",
         f"{int(sem_rotulo.sum())} instâncias sem rótulo pela lista "
         f"({int(df['genero_suspeito'].sum())} delas de gênero suspeito); "
         f"{int(sem_rotulo.sum()) - len(prov)} continuam sem rótulo após a amostra de negativos prováveis.\n",
         "## Casos ambíguos e suspeitos\n",
         f"{len(casos)} nomes (detalhes em `data/seeds/casos_ambiguos.csv`).\n",
         _tabela_md(casos["decisao"].value_counts().rename_axis("decisao").rename("instancias")),
         "## Correspondência entre a lista e o Names.txt\n",
         f"Gêneros da lista ausentes do Names.txt: {len(sem_corresp)} "
         "(lista completa em `results/generos_lista_sem_correspondencia.csv`).\n",
         _tabela_md(sem_corresp["ordem"].value_counts().rename_axis("ordem").rename("generos")),
         "Gêneros de Chiroptera da lista ausentes do Names.txt:\n",
         ", ".join(sorted(sem_corresp.loc[sem_corresp["ordem"] == ORDEM_MORCEGO, "genero"])) or "_(nenhum)_",
         "\n",
         f"No sentido Names.txt → lista: {generos_sem} gêneros sem correspondência "
         f"(não dá para saber quais são mamíferos sem a lista); "
         f"{via_sinonimo['genero'].nunique()} gêneros casados via sinônimo "
         f"({len(via_sinonimo)} instâncias).\n",
         ]
    return "\n".join(p)


# --------------------------------------------------------------------------
# Execução
# --------------------------------------------------------------------------

def executar(arq_taxonomia=ARQ_TAXONOMIA, arq_sinonimos=ARQ_SINONIMOS, dir_seeds=DIR_SEEDS,
             dir_resultados=DIR_RESULTADOS, n_negativos=300, fracao=0.2,
             seed=RANDOM_STATE, sobrescrever_teste=False, fracao_semente=0.35,
             max_especies_teste=9, min_positivos_teste=5):
    """Executa a Etapa 1 de ponta a ponta e devolve as tabelas geradas num dict.

    Levanta ``ListaTaxonomicaAusente`` (sem gravar nada) se a lista de gêneros
    não existir ou estiver vazia, e ``RuntimeError`` se o teste congelado
    fosse alterado sem ``sobrescrever_teste``.
    """
    if not Path(arq_taxonomia).exists():
        raise ListaTaxonomicaAusente(f"Arquivo não encontrado: {arq_taxonomia}")
    tax = carregar_taxonomia(arq_taxonomia)
    if tax.empty:
        raise ListaTaxonomicaAusente(
            f"{arq_taxonomia} só tem o cabeçalho. Coloque a lista real de gêneros de "
            "mamíferos (ver data/external/README.md) e rode de novo.")

    df, _ = carregar_tudo()
    df = anotar_taxonomia(df, tax, carregar_sinonimos(arq_sinonimos))
    n_especies = contar_especies(df)
    rotulados = montar_rotulados(df, montar_negativos_outros(df, n_negativos, seed))
    _, teste, excluidos, reservados = dividir_teste_por_genero(
        rotulados, n_especies, fracao, max_especies_teste, min_positivos_teste, seed)
    rotulados = dividir_semente_oculto(rotulados, reservados, fracao_semente, seed)
    teste = rotulados[rotulados["indice"].isin(teste["indice"])]
    semente = rotulados[rotulados["grupo"] == GRUPO_SEMENTE]
    gabarito = montar_gabarito(rotulados)
    bloqueados = montar_generos_bloqueados(rotulados, teste, reservados, n_especies)
    casos = montar_casos_ambiguos(df, rotulados, excluidos, reservados)
    sem_corresp = generos_lista_sem_correspondencia(tax, df)

    conteudo_teste = _csv_bytes(teste)
    verificar_congelamento(dir_seeds, conteudo_teste, sobrescrever_teste)
    salvar_sementes(dir_seeds, rotulados, semente, gabarito, bloqueados, conteudo_teste, casos)

    dir_resultados = Path(dir_resultados)
    dir_resultados.mkdir(parents=True, exist_ok=True)
    sem_corresp.to_csv(dir_resultados / "generos_lista_sem_correspondencia.csv", index=False)
    params = {"n_negativos": n_negativos, "fracao_teste_negativos": fracao,
              "fracao_semente": fracao_semente, "max_especies_teste": max_especies_teste,
              "min_positivos_teste": min_positivos_teste, "random_state": seed}
    relatorio = gerar_relatorio(df, tax, rotulados, semente, gabarito, teste, excluidos, bloqueados,
                                casos, sem_corresp, params)
    (dir_resultados / "relatorio_supervisao.md").write_text(relatorio, encoding="utf-8")

    return {"anotado": df, "rotulados": rotulados, "treino": semente, "semente": semente,
            "gabarito": gabarito, "teste": teste, "excluidos": excluidos,
            "bloqueados": bloqueados, "casos": casos, "n_especies": n_especies,
            "avisos": calcular_avisos(semente, gabarito, teste, min_positivos_teste)}


def main():
    parser = argparse.ArgumentParser(description="Supervisão inicial por gênero (Etapa 1).")
    parser.add_argument("--n-negativos", type=int, default=300,
                        help="tamanho da amostra de negativos prováveis (padrão: 300)")
    parser.add_argument("--fracao-semente", type=float, default=0.35,
                        help="fração dos gêneros fora do teste que entram na semente (padrão: 0.35)")
    parser.add_argument("--max-especies-teste", type=int, default=9,
                        help="gêneros com mais instâncias que isso nunca vão para o teste (padrão: 9)")
    parser.add_argument("--min-positivos-teste", type=int, default=5,
                        help="completa o teste com gêneros de morcego até esse nº de positivos (padrão: 5)")
    parser.add_argument("--sobrescrever-teste", action="store_true",
                        help="permite substituir o teste verificado congelado")
    args = parser.parse_args()
    try:
        tabelas = executar(n_negativos=args.n_negativos, sobrescrever_teste=args.sobrescrever_teste,
                           fracao_semente=args.fracao_semente, max_especies_teste=args.max_especies_teste,
                           min_positivos_teste=args.min_positivos_teste)
    except (ListaTaxonomicaAusente, RuntimeError) as erro:
        sys.exit(f"ERRO: {erro}")
    print(f"Sementes gravadas em {DIR_SEEDS}")
    print(f"Relatório em {DIR_RESULTADOS / 'relatorio_supervisao.md'}")
    for aviso in tabelas["avisos"]:
        print(f"AVISO: {aviso}")


if __name__ == "__main__":
    main()
