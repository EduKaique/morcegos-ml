"""Carga e validação dos dados brutos do trabalho "É morcego?".

Os arquivos ``data/raw/Names.txt`` e ``data/raw/wpc50.dat`` estão alinhados
pela posição: a linha *i* de um corresponde à linha *i* do outro. Nada aqui
reordena os registros; o número da linha original é a chave de ligação.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Base do índice usada em todo o projeto: 0 significa que a primeira linha
# dos arquivos tem indice == 0. Toda saída final deve informar qual base usa.
BASE_INDICE = 0

N_ESPERADO = 8426
N_COLUNAS = 50

DIR_RAIZ = Path(__file__).resolve().parents[1]
DIR_RAW = DIR_RAIZ / "data" / "raw"
ARQ_NOMES = "Names.txt"
ARQ_VETORES = "wpc50.dat"


def carregar_nomes(caminho):
    """Lê o arquivo de nomes científicos, um por linha.

    Remove ``\\r`` e espaços de cada linha e ignora linhas vazias.
    Linhas vazias no meio do arquivo quebrariam o alinhamento; nesse caso a
    contagem validada em ``carregar_tudo`` acusa o problema.

    Retorna uma lista de strings na ordem original do arquivo.
    """
    texto = Path(caminho).read_text(encoding="utf-8")
    return [linha.strip() for linha in texto.splitlines() if linha.strip()]


def carregar_vetores(caminho):
    """Lê a matriz de características (uma instância por linha).

    Retorna um ``np.ndarray`` de floats com formato (n_linhas, n_colunas),
    esperado (8426, 50).
    """
    return np.loadtxt(caminho, dtype=float, ndmin=2)


def separar_termos(nome):
    """Divide o nome em termos separados por ``_`` ou espaço.

    O padrão é ``_``, mas há nomes com espaço interno no arquivo
    (ex.: ``Crocodylus niloticus``), que precisam dar o mesmo resultado.
    """
    return re.split(r"[_\s]+", nome)


def extrair_genero(nome):
    """Retorna o gênero: o primeiro termo do nome (antes do ``_`` ou espaço)."""
    return separar_termos(nome)[0]


def _validar(nomes, X):
    """Levanta ``ValueError`` se nomes e vetores não estiverem consistentes."""
    if len(nomes) != N_ESPERADO:
        raise ValueError(f"{ARQ_NOMES}: esperados {N_ESPERADO} nomes, lidos {len(nomes)}.")
    if X.shape[0] != N_ESPERADO:
        raise ValueError(f"{ARQ_VETORES}: esperadas {N_ESPERADO} linhas, lidas {X.shape[0]}.")
    if X.shape[1] != N_COLUNAS:
        raise ValueError(f"{ARQ_VETORES}: esperadas {N_COLUNAS} colunas, lidas {X.shape[1]}.")
    if not np.isfinite(X).all():
        raise ValueError(f"{ARQ_VETORES}: há valores NaN ou infinitos.")


def carregar_tudo(dir_raw=DIR_RAW):
    """Carrega nomes e vetores, valida e devolve ``(df, X)``.

    ``df`` tem as colunas:
      - ``indice``: número da linha original na base ``BASE_INDICE`` (0);
      - ``indice_base1``: o mesmo número na base 1;
      - ``nome``: nome científico limpo;
      - ``genero``: primeiro termo do nome.

    ``X`` é o array (8426, 50); a linha ``X[i]`` corresponde a ``df.iloc[i]``.
    Levanta ``ValueError`` se a contagem, as colunas ou os valores falharem.
    """
    dir_raw = Path(dir_raw)
    nomes = carregar_nomes(dir_raw / ARQ_NOMES)
    X = carregar_vetores(dir_raw / ARQ_VETORES)
    _validar(nomes, X)

    indice = np.arange(len(nomes)) + BASE_INDICE
    df = pd.DataFrame({
        "indice": indice,
        "indice_base1": indice - BASE_INDICE + 1,
        "nome": nomes,
        "genero": [extrair_genero(n) for n in nomes],
    })
    return df, X


def _motivos_fora_do_padrao(nome):
    """Lista os motivos pelos quais um nome foge do padrão ``Genero_especie``."""
    termos = separar_termos(nome)
    termos_min = [t.lower().rstrip(".") for t in termos]
    motivos = []
    if len(termos) == 1:
        motivos.append("sem epiteto")
    if len(termos) > 2:
        motivos.append("mais de dois termos")
    if "sp" in termos_min:
        motivos.append("sp")
    if "nomen_nudum" in nome.lower():
        motivos.append("nomen_nudum")
    if "x" in termos_min:
        motivos.append("hibrido")
    if " " in nome:
        motivos.append("espaco interno")
    if not termos[0][:1].isupper():
        motivos.append("primeiro termo minusculo (genero suspeito)")
    return motivos


def listar_nomes_fora_do_padrao(df):
    """Devolve os registros cujo nome foge do padrão ``Genero_especie``.

    Motivos detectados: sem epíteto (um termo só), mais de dois termos
    (subespécies, linhagens etc.), ``sp``, ``nomen_nudum``, híbrido (``_x_``)
    espaço no lugar de ``_`` e primeiro termo em minúscula (ex.: ``diploid_...``),
    caso em que ``genero`` não é um gênero de verdade.
    Um nome pode ter vários motivos, separados por ``"; "`` na coluna ``motivo``.
    """
    motivos = df["nome"].map(_motivos_fora_do_padrao)
    fora = df.loc[motivos.str.len() > 0, ["indice", "nome", "genero"]].copy()
    fora["motivo"] = motivos[motivos.str.len() > 0].str.join("; ")
    return fora
