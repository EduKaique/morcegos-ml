import numpy as np
import pytest

from src.carregar_dados import (
    ARQ_NOMES,
    ARQ_VETORES,
    DIR_RAW,
    N_COLUNAS,
    N_ESPERADO,
    carregar_tudo,
)


@pytest.fixture(scope="module")
def dados():
    return carregar_tudo()


def test_8426_linhas(dados):
    df, X = dados
    assert len(df) == N_ESPERADO
    assert X.shape == (N_ESPERADO, N_COLUNAS)
    assert df["indice"].tolist() == list(range(N_ESPERADO))
    assert (df["indice_base1"] == df["indice"] + 1).all()


def test_alinhamento_com_arquivos_brutos(dados):
    """Confere, por leitura independente, que a linha i dos dois arquivos cai na posição i."""
    df, X = dados
    linhas_nomes = (DIR_RAW / ARQ_NOMES).read_bytes().split(b"\r\n")
    linhas_vetores = (DIR_RAW / ARQ_VETORES).read_bytes().split(b"\r\n")
    for i in [0, 8, 9, 4213, N_ESPERADO - 1]:
        assert df.loc[i, "nome"] == linhas_nomes[i].decode().strip()
        esperado = np.array(linhas_vetores[i].split(), dtype=float)
        np.testing.assert_array_equal(X[i], esperado)
    assert df.loc[0, "nome"] == "Candida_bracarensis"


def test_sem_nan(dados):
    _, X = dados
    assert not np.isnan(X).any()


def test_genero_e_primeiro_termo(dados):
    df, _ = dados
    assert (df["genero"] == df["nome"].str.split(r"[_\s]+", regex=True).str[0]).all()
    assert (df["nome"].str.len() > 0).all()
    assert (df["nome"] == df["nome"].str.strip()).all()
    assert not df["genero"].str.contains(r"[\s_]").any()


def test_erro_claro_com_arquivo_truncado(tmp_path):
    (tmp_path / ARQ_NOMES).write_text("Myotis_lucifugus\n")
    (tmp_path / ARQ_VETORES).write_text(" ".join(["0.1"] * N_COLUNAS) + "\n")
    with pytest.raises(ValueError, match="esperados 8426 nomes"):
        carregar_tudo(tmp_path)
