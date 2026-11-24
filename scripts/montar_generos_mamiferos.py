"""Converte o checklist do Mammal Diversity Database (MDD) em
data/external/generos_mamiferos.csv (colunas: genero,ordem,familia,fonte,data_acesso).

Uso:
    python scripts/montar_generos_mamiferos.py \
        --mdd data/external/MDD_v2.4.csv \
        --fonte "Mammal Diversity Database v2.4 (Zenodo)" \
        --data-acesso 2026-11-23 \
        --names data/raw/Names.txt

O MDD tem uma linha por espécie; aqui ficamos com uma linha por gênero.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

SAIDA_PADRAO = Path("data/external/generos_mamiferos.csv")
ALIASES = {
    "genero": ["genus"],
    "ordem": ["order"],
    "familia": ["family"],
}


def achar_coluna(df: pd.DataFrame, candidatos: list[str]) -> str:
    """Procura a coluna ignorando maiúsculas/minúsculas."""
    mapa = {c.strip().lower(): c for c in df.columns}
    for cand in candidatos:
        if cand in mapa:
            return mapa[cand]
    raise SystemExit(
        f"Coluna {candidatos} não encontrada. Colunas do arquivo: {list(df.columns)[:40]}"
    )


def montar_generos(mdd: pd.DataFrame, fonte: str, data_acesso: str) -> pd.DataFrame:
    col = {k: achar_coluna(mdd, v) for k, v in ALIASES.items()}
    df = mdd[[col["genero"], col["ordem"], col["familia"]]].copy()
    df.columns = ["genero", "ordem", "familia"]
    df = df.dropna()
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    df = df[df["genero"] != ""]
    # O MDD costuma escrever ordem e família em MAIÚSCULAS; padronizamos para
    # "Chiroptera", "Vespertilionidae" etc., pois o código compara com "Chiroptera".
    df["genero"] = df["genero"].str.capitalize()
    df["ordem"] = df["ordem"].str.capitalize()
    df["familia"] = df["familia"].str.capitalize()

    df = df.drop_duplicates()
    # Gênero com mais de uma família/ordem seria ambíguo: avisar e manter a mais frequente.
    conflitos = df.groupby("genero").size()
    conflitos = conflitos[conflitos > 1]
    if len(conflitos):
        print(f"AVISO: {len(conflitos)} gêneros com mais de uma ordem/família: "
              f"{list(conflitos.index)[:10]}", file=sys.stderr)
        contagem = (
            mdd.assign(genero=mdd[col["genero"]].astype(str).str.strip().str.capitalize())
            .groupby("genero").size()
        )
        df = df.assign(_n=df["genero"].map(contagem)).sort_values("_n", ascending=False)
        df = df.drop_duplicates("genero").drop(columns="_n")

    df["fonte"] = fonte
    df["data_acesso"] = data_acesso
    return df.sort_values(["ordem", "familia", "genero"]).reset_index(drop=True)


def relatar_cobertura(generos: pd.DataFrame, caminho_names: Path) -> None:
    """Mostra quantos nomes do Names.txt casam com a lista (conferência rápida)."""
    nomes = [
        linha.strip()
        for linha in caminho_names.read_text(encoding="ascii", errors="ignore").splitlines()
        if linha.strip()
    ]
    gen = pd.Series([n.replace(" ", "_").split("_")[0] for n in nomes])
    casados = gen.map(generos.set_index("genero")["ordem"])
    print(f"Instâncias do Names.txt: {len(nomes)}")
    print(f"  casadas com a lista (mamíferos): {casados.notna().sum()}")
    print(f"  morcegos (Chiroptera): {(casados == 'Chiroptera').sum()} instâncias em "
          f"{gen[casados == 'Chiroptera'].nunique()} gêneros")
    print("  mamíferos não-morcegos por ordem:")
    print(casados[casados != "Chiroptera"].value_counts().head(12).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mdd", required=True, type=Path, help="CSV do MDD baixado")
    ap.add_argument("--fonte", required=True, help="Ex.: 'Mammal Diversity Database v2.4 (Zenodo)'")
    ap.add_argument("--data-acesso", required=True, help="AAAA-MM-DD")
    ap.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    ap.add_argument("--names", type=Path, help="Names.txt, só para conferir a cobertura")
    args = ap.parse_args()

    mdd = pd.read_csv(args.mdd, low_memory=False)
    generos = montar_generos(mdd, args.fonte, args.data_acesso)
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    generos.to_csv(args.saida, index=False, encoding="utf-8", lineterminator="\n")

    print(f"Gerado {args.saida}: {len(generos)} gêneros, "
          f"{generos['ordem'].nunique()} ordens, "
          f"{(generos['ordem'] == 'Chiroptera').sum()} gêneros de Chiroptera.")
    if args.names:
        relatar_cobertura(generos, args.names)


if __name__ == "__main__":
    main()
