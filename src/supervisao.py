"""Supervisão inicial por gênero (Etapa 1).

Cruza o gênero de cada nome do ``Names.txt`` com uma lista externa de gêneros
de mamíferos (``data/external/generos_mamiferos.csv``), monta as sementes e
reserva gêneros inteiros para o teste verificado, que fica congelado por SHA256.

Uso: ``python -m src.supervisao`` a partir da raiz do repositório.
"""

import argparse
import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.carregar_dados import DIR_RAIZ, carregar_tudo, listar_nomes_fora_do_padrao

RANDOM_STATE = 42
ORDEM_MORCEGO = "Chiroptera"

DIR_EXTERNAL = DIR_RAIZ / "data" / "external"
DIR_SEEDS = DIR_RAIZ / "data" / "seeds"
DIR_RESULTADOS = DIR_RAIZ / "results"
ARQ_TAXONOMIA = DIR_EXTERNAL / "generos_mamiferos.csv"
ARQ_SINONIMOS = DIR_EXTERNAL / "sinonimos_generos.csv"

COLUNAS_TAXONOMIA = ["genero", "ordem", "familia", "fonte", "data_acesso"]
COLUNAS_SINONIMOS = ["genero_antigo", "genero_atual"]
COLUNAS_SAIDA = ["indice", "nome", "genero", "genero_atual", "ordem", "familia", "rotulo", "origem"]

# Motivos de listar_nomes_fora_do_padrao que tornam um nome ambíguo.
MOTIVOS_AMBIGUOS = {"sp", "nomen_nudum", "hibrido", "mais de dois termos"}
MOTIVO_SUSPEITO = "primeiro termo minusculo"

ORIGEM_MORCEGO = "chiroptera_por_genero"
ORIGEM_MAMIFERO = "mamifero_por_genero"
ORIGEM_PROVAVEL = "negativo_provavel"


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


def familias_elegiveis(df_rotulado):
    """Famílias de morcego com 2+ gêneros elegíveis (com algum nome não ambíguo)."""
    morcegos = df_rotulado[(df_rotulado["origem"] == ORIGEM_MORCEGO) & ~df_rotulado["ambiguo"]]
    n_generos = morcegos.groupby("familia")["genero_atual"].nunique()
    return sorted(n_generos[n_generos >= 2].index)


def dividir_teste_por_genero(df_rotulado, fracao=0.2, seed=RANDOM_STATE):
    """Reserva gêneros inteiros para o teste verificado. Devolve ``(treino, teste, excluidos)``.

    Regras (determinísticas para o mesmo ``seed``):
      - só gêneros com ao menos um nome não ambíguo podem ser reservados;
      - morcegos: 1 gênero por família com 2+ gêneros elegíveis (família com um
        gênero só fica inteira no treino);
      - mamíferos não-morcegos: 1 gênero por ordem com 2+ gêneros (cobre
        Eulipotyphla e demais insetívoros, se presentes), completando até
        ``fracao`` dos gêneros;
      - negativos prováveis: ``fracao`` dos gêneros;
      - nos gêneros reservados, os nomes não ambíguos vão para o teste e os
        ambíguos vão para ``excluidos`` (fora do teste e do treino, para não
        vazar o gênero).
    O grupo é ``genero_atual``.
    """
    rng = np.random.default_rng(seed)
    elegiveis = df_rotulado[~df_rotulado["ambiguo"]]
    por_origem = {o: elegiveis[elegiveis["origem"] == o] for o in
                  (ORIGEM_MORCEGO, ORIGEM_MAMIFERO, ORIGEM_PROVAVEL)}

    reservados = _um_por_grupo(rng, por_origem[ORIGEM_MORCEGO], "familia")

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
    treino = df_rotulado[~no_teste]
    return treino, teste, excluidos


# --------------------------------------------------------------------------
# Casos ambíguos, gravação e congelamento
# --------------------------------------------------------------------------

def montar_casos_ambiguos(df, rotulados, excluidos):
    """Tabela de nomes ambíguos e suspeitos com a decisão tomada para cada um.

    A coluna ``observacao`` fica vazia para a equipe preencher.
    """
    casos = df[df["ambiguo"] | df["genero_suspeito"]].copy()
    provaveis = set(rotulados.loc[rotulados["origem"] == ORIGEM_PROVAVEL, "indice"])
    casos["decisao"] = np.select(
        [casos["genero_suspeito"],
         casos["indice"].isin(excluidos["indice"]),
         casos["indice"].isin(provaveis),
         casos["rotulo_fonte"].notna()],
        ["sem rotulo (genero suspeito)",
         "excluido (genero reservado ao teste)",
         "negativo provavel; fora do teste",
         "rotulado por genero; fora do teste"],
        default="sem rotulo (genero fora da lista)",
    )
    casos["observacao"] = ""
    colunas = ["indice", "nome", "genero", "genero_atual", "motivo", "rotulo_fonte", "decisao", "observacao"]
    return casos[colunas]


def _csv_bytes(tabela, colunas=COLUNAS_SAIDA):
    """Serializa a tabela em CSV (UTF-8, quebra de linha LF) ordenada por indice."""
    tabela = tabela.sort_values("indice")[colunas]
    return tabela.to_csv(index=False, lineterminator="\n").encode("utf-8")


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


def salvar_sementes(dir_seeds, rotulados, treino, conteudo_teste, casos):
    """Grava positivos, negativos, treino, teste, casos ambíguos e o SHA256 do teste."""
    dir_seeds = Path(dir_seeds)
    dir_seeds.mkdir(parents=True, exist_ok=True)
    (dir_seeds / "positivos.csv").write_bytes(_csv_bytes(rotulados[rotulados["rotulo"] == 1]))
    (dir_seeds / "negativos.csv").write_bytes(_csv_bytes(rotulados[rotulados["rotulo"] == 0]))
    (dir_seeds / "treino_inicial.csv").write_bytes(_csv_bytes(treino))
    (dir_seeds / "teste_verificado.csv").write_bytes(conteudo_teste)
    (dir_seeds / "casos_ambiguos.csv").write_bytes(_csv_bytes(casos, list(casos.columns)))
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


def generos_lista_sem_correspondencia(tax, df):
    """Gêneros da lista externa que não aparecem no Names.txt (após sinônimos)."""
    return tax[~tax["genero"].isin(set(df["genero_atual"]))][["genero", "ordem", "familia"]]


def gerar_relatorio(df, tax, rotulados, treino, teste, excluidos, casos, sem_corresp, n_pedido):
    """Monta o texto Markdown do relatório da supervisão."""
    pos = rotulados[rotulados["origem"] == ORIGEM_MORCEGO]
    mam = rotulados[rotulados["origem"] == ORIGEM_MAMIFERO]
    prov = rotulados[rotulados["origem"] == ORIGEM_PROVAVEL]
    sem_rotulo = df["rotulo_fonte"].isna()
    casados = df[df["rotulo_fonte"].notna()]
    via_sinonimo = casados[casados["genero"] != casados["genero_atual"]]
    generos_sem = df.loc[sem_rotulo & ~df["genero_suspeito"], "genero_atual"].nunique()
    fontes = ", ".join(sorted(set(tax["fonte"]) - {""})) or "(não informada)"

    p = ["# Relatório da supervisão inicial (Etapa 1)\n",
         f"Lista taxonômica: {len(tax)} gêneros; fonte(s): {fontes}. "
         f"Índices na base 0 (`BASE_INDICE`).\n",
         "## Positivos (Chiroptera) por família\n",
         f"Total: {len(pos)} instâncias de {pos['genero_atual'].nunique()} gêneros.\n",
         _tabela_md(_contagem(pos, "familia")),
         "## Negativos mamíferos por ordem\n",
         f"Total: {len(mam)} instâncias de {mam['genero_atual'].nunique()} gêneros.\n",
         _tabela_md(_contagem(mam, "ordem")),
         "## Negativos prováveis\n",
         f"{len(prov)} instâncias (pedido: {n_pedido}) de {prov['genero_atual'].nunique()} gêneros, "
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
         "## Teste verificado\n",
         f"{len(teste)} instâncias de {teste['genero_atual'].nunique()} gêneros inteiros "
         f"(positivos: {int((teste['rotulo'] == 1).sum())}, negativos: {int((teste['rotulo'] == 0).sum())}). "
         f"{len(excluidos)} nomes ambíguos de gêneros reservados ficaram fora do teste e do treino.\n",
         _tabela_md(_contagem(teste, ["origem", "ordem", "familia"]).fillna("-")),
         "## Treino inicial\n",
         f"{len(treino)} instâncias de {treino['genero_atual'].nunique()} gêneros "
         f"(positivos: {int((treino['rotulo'] == 1).sum())}, negativos: {int((treino['rotulo'] == 0).sum())}).\n",
         _tabela_md(_contagem(treino, "origem")),
         ]
    return "\n".join(p)


# --------------------------------------------------------------------------
# Execução
# --------------------------------------------------------------------------

def executar(arq_taxonomia=ARQ_TAXONOMIA, arq_sinonimos=ARQ_SINONIMOS, dir_seeds=DIR_SEEDS,
             dir_resultados=DIR_RESULTADOS, n_negativos=300, fracao=0.2,
             seed=RANDOM_STATE, sobrescrever_teste=False):
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
    rotulados = montar_rotulados(df, montar_negativos_outros(df, n_negativos, seed))
    treino, teste, excluidos = dividir_teste_por_genero(rotulados, fracao, seed)
    casos = montar_casos_ambiguos(df, rotulados, excluidos)
    sem_corresp = generos_lista_sem_correspondencia(tax, df)

    conteudo_teste = _csv_bytes(teste)
    verificar_congelamento(dir_seeds, conteudo_teste, sobrescrever_teste)
    salvar_sementes(dir_seeds, rotulados, treino, conteudo_teste, casos)

    dir_resultados = Path(dir_resultados)
    dir_resultados.mkdir(parents=True, exist_ok=True)
    sem_corresp.to_csv(dir_resultados / "generos_lista_sem_correspondencia.csv", index=False)
    relatorio = gerar_relatorio(df, tax, rotulados, treino, teste, excluidos, casos, sem_corresp, n_negativos)
    (dir_resultados / "relatorio_supervisao.md").write_text(relatorio, encoding="utf-8")

    return {"anotado": df, "rotulados": rotulados, "treino": treino, "teste": teste,
            "excluidos": excluidos, "casos": casos}


def main():
    parser = argparse.ArgumentParser(description="Supervisão inicial por gênero (Etapa 1).")
    parser.add_argument("--n-negativos", type=int, default=300,
                        help="tamanho da amostra de negativos prováveis (padrão: 300)")
    parser.add_argument("--sobrescrever-teste", action="store_true",
                        help="permite substituir o teste verificado congelado")
    args = parser.parse_args()
    try:
        executar(n_negativos=args.n_negativos, sobrescrever_teste=args.sobrescrever_teste)
    except (ListaTaxonomicaAusente, RuntimeError) as erro:
        sys.exit(f"ERRO: {erro}")
    print(f"Sementes gravadas em {DIR_SEEDS}")
    print(f"Relatório em {DIR_RESULTADOS / 'relatorio_supervisao.md'}")


if __name__ == "__main__":
    main()
