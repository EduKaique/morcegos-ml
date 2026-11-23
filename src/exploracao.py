"""Exploração inicial dos dados (Etapa 0). Nenhum rótulo é criado aqui.

Uso: ``python -m src.exploracao`` a partir da raiz do repositório.
Gera tabelas em ``results/`` e figuras em ``results/figuras/``.
"""

import matplotlib

matplotlib.use("Agg")  # roda sem janela gráfica

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

from src.carregar_dados import DIR_RAIZ, carregar_tudo, listar_nomes_fora_do_padrao

RANDOM_STATE = 42
DIR_RESULTADOS = DIR_RAIZ / "results"
DIR_FIGURAS = DIR_RESULTADOS / "figuras"


def estatisticas_descritivas(X):
    """Retorna ``describe()`` das 50 características, uma linha por coluna."""
    colunas = [f"c{i:02d}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=colunas).describe().T


def contar_especies_por_genero(df):
    """Retorna uma Series gênero -> nº de espécies, em ordem decrescente."""
    return df["genero"].value_counts()


def quebras_de_ordem(df):
    """Retorna os registros cujo nome é alfabeticamente menor que o anterior."""
    nomes = df["nome"].str.lower()
    return df.loc[nomes < nomes.shift(1), ["indice", "nome"]]


def salvar_dispersao(Y, titulo, caminho):
    """Salva um gráfico de dispersão 2D sem cores de classe."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(Y[:, 0], Y[:, 1], s=3, alpha=0.5, color="steelblue", linewidths=0)
    ax.set_title(titulo)
    ax.set_xlabel("componente 1")
    ax.set_ylabel("componente 2")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def projetar_pca(X):
    """Projeta X em 2D com PCA; retorna (Y, variância explicada por componente)."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    return pca.fit_transform(X), pca.explained_variance_ratio_


def projetar_umap(X):
    """Projeta X em 2D com UMAP (random_state fixo)."""
    import umap  # importação tardia: o umap demora a carregar

    return umap.UMAP(n_components=2, random_state=RANDOM_STATE).fit_transform(X)


def salvar_histograma_generos(contagem, caminho):
    """Salva o histograma de nº de espécies por gênero (eixo y em log)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = range(1, contagem.max() + 2)
    ax.hist(contagem.values, bins=bins, color="steelblue", edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("espécies por gênero")
    ax.set_ylabel("nº de gêneros (log)")
    ax.set_title(f"Espécies por gênero ({len(contagem)} gêneros)")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def main():
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    df, X = carregar_tudo()
    print(f"Registros: {len(df)} | X: {X.shape} | nomes repetidos: {df['nome'].duplicated().sum()}")

    desc = estatisticas_descritivas(X)
    desc.to_csv(DIR_RESULTADOS / "estatisticas_descritivas.csv")
    print(f"\nMédia global: {X.mean():.4f} | desvio global: {X.std():.4f}")
    print(f"Média por coluna: [{desc['mean'].min():.4f}, {desc['mean'].max():.4f}]")
    print(f"Desvio por coluna: [{desc['std'].min():.4f}, {desc['std'].max():.4f}]")

    contagem = contar_especies_por_genero(df)
    print(f"\nGêneros distintos: {len(contagem)} | gêneros com 1 espécie: {(contagem == 1).sum()}")
    print("Top 20 gêneros por nº de espécies:")
    print(contagem.head(20).to_string())

    fora = listar_nomes_fora_do_padrao(df)
    fora.to_csv(DIR_RESULTADOS / "nomes_fora_do_padrao.csv", index=False)
    print(f"\nNomes fora do padrão: {len(fora)}")
    print(fora["motivo"].str.split("; ").explode().value_counts().to_string())

    quebras = quebras_de_ordem(df)
    print(f"\nQuebras da ordem alfabética: {len(quebras)}")
    print(quebras.to_string(index=False))

    Y_pca, var = projetar_pca(X)
    salvar_dispersao(Y_pca, f"PCA 2D (variância explicada: {var[0]:.1%} + {var[1]:.1%})",
                     DIR_FIGURAS / "pca_2d.png")
    salvar_dispersao(projetar_umap(X), "UMAP 2D (random_state=42)", DIR_FIGURAS / "umap_2d.png")
    salvar_histograma_generos(contagem, DIR_FIGURAS / "hist_especies_por_genero.png")
    print(f"\nFiguras salvas em {DIR_FIGURAS}")


if __name__ == "__main__":
    main()
