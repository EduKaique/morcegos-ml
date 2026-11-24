"""Testes da Etapa 5 (relatório das classes finais).

Usam as entradas reais do repositório (``data/seeds``, ``data/rounds``,
``results/classes_finais.csv`` etc.) só para leitura; toda saída vai para
``tmp_path``. Se as classes finais ainda não foram geradas, os testes são pulados.
"""

import re
import shutil

import pandas as pd
import pytest

import src.exploracao
import src.relatorio_etapa5 as r5
from src.carregar_dados import DIR_RAIZ

RODADA = 3
DIR_RES = DIR_RAIZ / "results"
ENTRADAS = (r5.ARQ_CLASSES, r5.ARQ_LISTA, r5.ARQ_COMPARACAO)

pytestmark = pytest.mark.skipif(
    not all((DIR_RES / a).exists() for a in ENTRADAS)
    or not (DIR_RAIZ / "data" / "rounds" / f"rodada_{RODADA:02d}" / "base.csv").exists(),
    reason="classes finais / rodada manual ainda não geradas")


def _proibir_umap(*_a, **_k):
    raise AssertionError("UMAP não pode ser calculado na Etapa 5")


def _copiar_entradas(destino):
    destino.mkdir(parents=True, exist_ok=True)
    for a in ENTRADAS:
        shutil.copy(DIR_RES / a, destino / a)
    return destino


def _executar(dir_entrada, dir_saida, monkeypatch):
    monkeypatch.setattr(src.exploracao, "projetar_umap", _proibir_umap)
    return r5.executar(RODADA, 1, dir_entrada=dir_entrada, dir_saida=dir_saida)


@pytest.fixture(scope="module")
def execucao(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    try:
        base = tmp_path_factory.mktemp("etapa5")
        dir_saida = base / "saida"
        r = _executar(_copiar_entradas(base / "entrada"), dir_saida, mp)
    finally:
        mp.undo()
    return r, dir_saida


# --------------------------------------------------------------------------
# Verificações de consistência
# --------------------------------------------------------------------------

def test_saidas_geradas_sem_umap_novo(execucao):
    r, dir_saida = execucao
    assert (dir_saida / r5.ARQ_MORCEGOS).exists() and (dir_saida / r5.ARQ_RELATORIO).exists()
    for f in r5.FIGURAS:
        assert (dir_saida / "figuras" / f).exists()
    # sem cache na entrada temporária: o UMAP é pulado e registrado
    assert not r["umap_ok"]
    assert not (dir_saida / "figuras" / r5.FIGURA_UMAP).exists()
    assert "UMAP não gerado" in (dir_saida / r5.ARQ_RELATORIO).read_text(encoding="utf-8")


@pytest.mark.parametrize("adulterar, mensagem", [
    (lambda c: c.drop(index=10), "linhas"),
    (lambda c: c.assign(classe=c["classe"].where(c.index != 0, 2)), "classes fora"),
    (lambda c: c.assign(indice_base0=c["indice_base0"].where(c.index != 1, 0)), "repetidos"),
    (lambda c: c.assign(classe=c["classe"].where(c.index != 0, 1)), "indices_positivos"),
])
def test_consistencia_falha_ao_adulterar_classes(tmp_path, adulterar, mensagem, monkeypatch):
    dir_ent = _copiar_entradas(tmp_path / "entrada")
    arq = dir_ent / r5.ARQ_CLASSES
    adulterar(pd.read_csv(arq)).to_csv(arq, index=False)
    with pytest.raises(r5.ErroConsistencia, match=mensagem):
        _executar(dir_ent, tmp_path / "saida", monkeypatch)
    assert not (tmp_path / "saida" / r5.ARQ_RELATORIO).exists()


def test_consistencia_falha_com_lista_na_base_errada(tmp_path, monkeypatch):
    dir_ent = _copiar_entradas(tmp_path / "entrada")
    lista = r5.ler_lista(dir_ent / r5.ARQ_LISTA)
    (dir_ent / r5.ARQ_LISTA).write_text("\n".join(str(i - 1) for i in lista) + "\n", encoding="utf-8")
    with pytest.raises(r5.ErroConsistencia, match="base 0"):
        _executar(dir_ent, tmp_path / "saida", monkeypatch)


def test_consistencia_falha_com_positivo_fora_do_gabarito(tmp_path, monkeypatch):
    dir_ent = _copiar_entradas(tmp_path / "entrada")
    arq = dir_ent / r5.ARQ_CLASSES
    c = pd.read_csv(arq)
    c.loc[c.index[c["classe"] == 1][0], "fonte_rotulo"] = "modelo_final"
    c.to_csv(arq, index=False)
    with pytest.raises(r5.ErroConsistencia, match="não vêm do gabarito"):
        _executar(dir_ent, tmp_path / "saida", monkeypatch)


def test_verificar_consistencia_detecta_bloqueado_nos_candidatos(execucao):
    c = pd.read_csv(DIR_RES / r5.ARQ_CLASSES)
    lista = r5.ler_lista(DIR_RES / r5.ARQ_LISTA)
    vazio = pd.DataFrame({"indice": []})
    with pytest.raises(r5.ErroConsistencia, match="bloqueados em candidatos"):
        r5.verificar_consistencia(c, lista, 1, {5}, pd.DataFrame({"indice": [5]}), vazio)
    tab = r5.verificar_consistencia(c, lista, 1, {5}, vazio, vazio)
    assert int(tab.to_numpy().sum()) == len(c)


# --------------------------------------------------------------------------
# Morcegos e relatório
# --------------------------------------------------------------------------

def test_origem_entrada_cobre_todos_os_morcegos(execucao):
    r, dir_saida = execucao
    mor = pd.read_csv(dir_saida / r5.ARQ_MORCEGOS)
    classes = pd.read_csv(DIR_RES / r5.ARQ_CLASSES)
    assert sorted(mor["indice_base0"]) == sorted(classes.loc[classes["classe"] == 1, "indice_base0"])
    assert (mor["indice_base1"] == mor["indice_base0"] + 1).all()
    assert mor["origem_entrada"].notna().all()
    assert set(mor["origem_entrada"]) <= set(r5.origens_validas(RODADA))
    assert (mor["genero"] != "").all() and (mor["familia"] != "").all()


def test_contagens_do_relatorio_batem_com_csv(execucao):
    _, dir_saida = execucao
    texto = (dir_saida / r5.ARQ_RELATORIO).read_text(encoding="utf-8")
    classes = pd.read_csv(DIR_RES / r5.ARQ_CLASSES)
    r5.verificar_contagens_relatorio(texto, classes)
    adulterado = re.sub(r"^\| modelo_final \| (\d+) \|", lambda m: f"| modelo_final | {int(m.group(1)) + 1} |",
                        texto, flags=re.M)
    with pytest.raises(r5.ErroConsistencia, match="modelo_final"):
        r5.verificar_contagens_relatorio(adulterado, classes)


def test_relatorio_sem_caminho_absoluto(execucao):
    _, dir_saida = execucao
    texto = (dir_saida / r5.ARQ_RELATORIO).read_text(encoding="utf-8")
    for proibido in (str(DIR_RAIZ), DIR_RAIZ.as_posix(), str(dir_saida), dir_saida.as_posix()):
        assert proibido not in texto
    assert not re.search(r"[A-Za-z]:[\\/]", texto)
    assert "/home/" not in texto and "/Users/" not in texto


def test_umap_em_cache_com_hash_errado_e_ignorado(tmp_path):
    import numpy as np
    np.save(tmp_path / r5.ARQ_UMAP_CACHE, np.zeros((5, 2)))
    assert r5.carregar_umap_cache(tmp_path)[0] is None  # sem arquivo de hash
    (tmp_path / r5.ARQ_UMAP_HASH).write_text("0" * 64, encoding="utf-8")
    Y, motivo = r5.carregar_umap_cache(tmp_path)
    assert Y is None and "não confere" in motivo


def test_reprodutivel(execucao, tmp_path, monkeypatch):
    r1, dir1 = execucao
    r2 = _executar(_copiar_entradas(tmp_path / "entrada"), tmp_path / "saida", monkeypatch)
    for arq in (r5.ARQ_MORCEGOS, r5.ARQ_RELATORIO):
        assert (dir1 / arq).read_bytes() == (tmp_path / "saida" / arq).read_bytes()
    for m in ("svm", "rf"):
        pd.testing.assert_frame_equal(r1["segunda"][m]["top"], r2["segunda"][m]["top"])
