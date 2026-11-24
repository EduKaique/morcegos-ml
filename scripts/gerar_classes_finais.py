"""Etapa 5: gera as 8426 classes finais (1 = morcego, 0 = outro) e a lista de índices dos morcegos.

Prioridade do rótulo de cada instância:
  1. gabarito taxonômico (data/seeds/gabarito.csv)
  2. teste verificado    (data/seeds/teste_verificado.csv)
  3. base final          (data/rounds/rodada_NN/base.csv: semente + decisões humanas)
  4. modelo final        (results/etapa4_pontuacoes_todas.csv, pontuação >= limiar)

Se duas fontes verificadas discordarem sobre o mesmo índice, o script para com erro.
Os índices dos arquivos são base 0; a saída traz base 0 e base 1.

Uso (na raiz do repositório, com o ambiente ativado):
    python scripts/gerar_classes_finais.py --rodada-final 3 --base-lista 1
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

N_TOTAL = 8426
LIMIAR_PADRAO = 0.6728  # fixado na Etapa 2 (results/relatorio_etapa2.md)


def ler(caminho: Path, colunas: list[str]) -> pd.DataFrame:
    if not caminho.exists():
        sys.exit(f"Arquivo não encontrado: {caminho}")
    df = pd.read_csv(caminho)
    faltam = [c for c in colunas if c not in df.columns]
    if faltam:
        sys.exit(f"{caminho} sem as colunas {faltam}. Colunas existentes: {list(df.columns)}")
    return df


def rotulos_verificados(gabarito: pd.DataFrame, teste: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Une as três fontes verificadas, na ordem de prioridade, e checa conflitos."""
    partes = [
        gabarito[["indice", "rotulo_gabarito"]].rename(columns={"rotulo_gabarito": "rotulo"}).assign(fonte="gabarito"),
        teste[["indice", "rotulo"]].assign(fonte="teste_verificado"),
        base[["indice", "rotulo"]].assign(fonte="base_final"),
    ]
    todas = pd.concat(partes, ignore_index=True).dropna(subset=["rotulo"])
    todas["rotulo"] = todas["rotulo"].astype(int)

    conflito = todas.groupby("indice")["rotulo"].nunique()
    conflito = conflito[conflito > 1]
    if len(conflito):
        sys.exit(f"Conflito de rótulo entre fontes para {len(conflito)} índices: {list(conflito.index)[:10]}")

    # mantém a primeira fonte na ordem de prioridade
    return todas.drop_duplicates("indice", keep="first").set_index("indice")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rodada-final", type=int, required=True)
    ap.add_argument("--limiar", type=float, default=LIMIAR_PADRAO)
    ap.add_argument("--base-lista", type=int, choices=[0, 1], default=1,
                    help="base dos índices em indices_positivos.txt (1 = número de ordem a partir de 1)")
    ap.add_argument("--saida", type=Path, default=Path("results"))
    args = ap.parse_args()

    gab = ler(Path("data/seeds/gabarito.csv"), ["indice", "rotulo_gabarito"])
    teste = ler(Path("data/seeds/teste_verificado.csv"), ["indice", "rotulo"])
    base = ler(Path(f"data/rounds/rodada_{args.rodada_final:02d}/base.csv"), ["indice", "rotulo"])
    pont = ler(Path("results/etapa4_pontuacoes_todas.csv"), ["indice", "nome", "pontuacao"])

    if len(pont) != N_TOTAL or pont["indice"].nunique() != N_TOTAL:
        sys.exit(f"etapa4_pontuacoes_todas.csv deveria ter {N_TOTAL} índices únicos")

    verif = rotulos_verificados(gab, teste, base)

    df = pont[["indice", "nome", "pontuacao"]].copy()
    df["classe"] = df["indice"].map(verif["rotulo"])
    df["fonte_rotulo"] = df["indice"].map(verif["fonte"])

    sem_rotulo = df["classe"].isna()
    df.loc[sem_rotulo, "classe"] = (df.loc[sem_rotulo, "pontuacao"] >= args.limiar).astype(int)
    df.loc[sem_rotulo, "fonte_rotulo"] = "modelo_final"
    df["classe"] = df["classe"].astype(int)

    df["indice_base0"] = df["indice"]
    df["indice_base1"] = df["indice"] + 1
    df = df.sort_values("indice").reset_index(drop=True)

    # verificações
    assert len(df) == N_TOTAL and df["indice"].is_unique
    assert set(df["classe"].unique()) <= {0, 1}

    args.saida.mkdir(parents=True, exist_ok=True)
    df[["indice_base0", "indice_base1", "nome", "classe", "fonte_rotulo", "pontuacao"]].to_csv(
        args.saida / "classes_finais.csv", index=False)
    positivos = df.loc[df["classe"] == 1, f"indice_base{args.base_lista}"]
    (args.saida / "indices_positivos.txt").write_text("\n".join(map(str, positivos)) + "\n", encoding="utf-8")

    # resumo e pontos que exigem olhar humano
    print(f"Classes finais: {len(df)} instâncias, {int(df['classe'].sum())} morcegos (índices em base {args.base_lista}).")
    print("\nFonte do rótulo x classe:")
    print(pd.crosstab(df["fonte_rotulo"], df["classe"]).to_string())
    pos_modelo = df[(df["fonte_rotulo"] == "modelo_final") & (df["classe"] == 1)]
    print(f"\nPositivos decididos SÓ pelo modelo (conferir um a um na taxonomia): {len(pos_modelo)}")
    if len(pos_modelo):
        print(pos_modelo[["indice_base0", "nome", "pontuacao"]].to_string(index=False))
    discordam = df[(df["fonte_rotulo"] != "modelo_final") & ((df["pontuacao"] >= args.limiar) != (df["classe"] == 1))]
    print(f"\nInstâncias em que o rótulo verificado discorda do modelo: {len(discordam)}")
    if len(discordam):
        print(discordam[["indice_base0", "nome", "classe", "pontuacao", "fonte_rotulo"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
