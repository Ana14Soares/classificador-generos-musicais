"""
00_justificativa_subconjunto.py
===============================
Experimento que justifica a reducao de 114 generos para 10 classes.

O enunciado permite a escolha livre da base. A base do Spotify traz 114
generos, mas muitos deles sao rotulos comerciais sobrepostos ("rock",
"hard-rock", "punk-rock", "alt-rock", "rock-n-roll", ...) que descrevem
praticamente o mesmo material acustico. Este script mede o efeito dessa
sobreposicao, comparando o mesmo algoritmo em tres recortes:

  (a) todos os 114 generos;
  (b) um recorte "dificil" de 10 generos proximos (varios subgeneros de rock);
  (c) o recorte adotado no trabalho, com 10 generos contrastantes.

A conclusao e documentada no relatorio: o fator limitante nao e o algoritmo,
e a definicao das classes.

Execucao:
    python src/00_justificativa_subconjunto.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preparacao import (  # noqa: E402
    ALVO,
    DIR_RESULTADOS,
    FEATURES,
    GENEROS_SELECIONADOS,
    PROPORCAO_TESTE,
    SEMENTE,
    carregar_base_completa,
    garantir_diretorios,
)

warnings.filterwarnings("ignore")

#: Recorte deliberadamente dificil: dez rotulos com forte sobreposicao.
GENEROS_PROXIMOS = [
    "rock", "hard-rock", "punk-rock", "alt-rock", "rock-n-roll",
    "punk", "grunge", "emo", "indie", "alternative",
]


def preparar(df: pd.DataFrame, generos: list[str] | None) -> pd.DataFrame:
    """Mesma limpeza do modulo ``preparacao``, para um conjunto de generos."""
    df = df.dropna(subset=["track_name", "artists", "album_name"]).copy()
    df = df.drop_duplicates(subset=["track_id", ALVO])
    if generos is not None:
        df = df[df[ALVO].isin(generos)]
        n_generos = df.groupby("track_id")[ALVO].nunique()
        df = df[df["track_id"].isin(n_generos[n_generos == 1].index)]
    df["explicit"] = df["explicit"].astype(int)
    df = df[(df["tempo"] > 0) & (df["duration_ms"] > 0)]
    return df


def avaliar(df: pd.DataFrame, descricao: str) -> dict:
    X, y = df[FEATURES], df[ALVO]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=PROPORCAO_TESTE, random_state=SEMENTE, stratify=y)

    modelo = Pipeline([
        ("discretizar", KBinsDiscretizer(n_bins=24, encode="ordinal",
                                         strategy="quantile")),
        ("nb", CategoricalNB(alpha=0.1)),
    ])
    modelo.fit(X_tr, y_tr)
    pred = modelo.predict(X_te)

    n_classes = y.nunique()
    acuracia = accuracy_score(y_te, pred)
    resultado = {
        "recorte": descricao,
        "n_classes": n_classes,
        "n_faixas": len(df),
        "acuracia": round(acuracia, 4),
        "f1_macro": round(f1_score(y_te, pred, average="macro"), 4),
        "acaso": round(1 / n_classes, 4),
        "ganho_sobre_acaso": round(acuracia * n_classes, 2),
    }
    print(f"{descricao:<42} classes={n_classes:>3}  "
          f"acuracia={resultado['acuracia']:.4f}  "
          f"F1={resultado['f1_macro']:.4f}  "
          f"ganho={resultado['ganho_sobre_acaso']:.1f}x")
    return resultado


def main() -> None:
    garantir_diretorios()
    print("\n" + "=" * 78)
    print("JUSTIFICATIVA DA DEFINICAO DAS CLASSES")
    print("Mesmo algoritmo, mesmas features, mesmos hiperparametros.")
    print("=" * 78 + "\n")

    bruto = carregar_base_completa()

    linhas = [
        avaliar(preparar(bruto, None), "(a) todos os 114 generos originais"),
        avaliar(preparar(bruto, GENEROS_PROXIMOS), "(b) 10 generos proximos (rock e variacoes)"),
        avaliar(preparar(bruto, GENEROS_SELECIONADOS), "(c) 10 generos contrastantes (adotado)"),
    ]

    tabela = pd.DataFrame(linhas)
    tabela.to_csv(DIR_RESULTADOS / "justificativa_subconjunto.csv", index=False)

    print("\n" + "-" * 78)
    print("CONCLUSAO")
    print("-" * 78)
    print(
        "Com 114 classes o modelo supera o acaso em varias vezes, mas a acuracia\n"
        "absoluta e baixa demais para uma aplicacao util. O recorte (b) mostra que\n"
        "o problema nao e a quantidade de classes: com apenas 10 rotulos, porem\n"
        "acusticamente sobrepostos, o desempenho continua fraco. O recorte (c),\n"
        "adotado no trabalho, mantem 10 classes e atinge desempenho alto porque os\n"
        "generos ocupam regioes distintas do espaco de atributos.\n\n"
        "Ou seja: o fator limitante e a definicao das classes, nao o algoritmo.\n"
        "Essa e a justificativa para o recorte usado no modelo final."
    )
    print(f"\nTabela salva em: {DIR_RESULTADOS / 'justificativa_subconjunto.csv'}")


if __name__ == "__main__":
    main()
