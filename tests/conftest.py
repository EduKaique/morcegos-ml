from pathlib import Path

import pandas as pd
import pytest

from src.supervisao import ARQ_SINONIMOS, ARQ_TAXONOMIA, executar

DIR_FIXTURES = Path(__file__).parent / "fixtures"


def _tem_linhas(caminho):
    return caminho.exists() and len(pd.read_csv(caminho)) > 0


def caminhos_taxonomia():
    """Lista real se ela tiver linhas; senão, o fixture (não verificado)."""
    if _tem_linhas(ARQ_TAXONOMIA):
        return ARQ_TAXONOMIA, ARQ_SINONIMOS
    return DIR_FIXTURES / "generos_mamiferos_teste.csv", DIR_FIXTURES / "sinonimos_generos_teste.csv"


@pytest.fixture(scope="session")
def taxonomia():
    return caminhos_taxonomia()


@pytest.fixture(scope="session")
def execucao(tmp_path_factory, taxonomia):
    """Roda a Etapa 1 inteira em diretórios temporários (nunca em data/seeds/)."""
    base = tmp_path_factory.mktemp("supervisao")
    arq_tax, arq_sin = taxonomia
    tabelas = executar(arq_tax, arq_sin, base / "seeds", base / "results")
    return {"dir": base, **tabelas}
