"""
01_analise_exploratoria.py
==========================
Analise Exploratoria de Dados (AED) da base "Spotify Tracks Dataset".

Gera, de forma reproduzivel:
  * caracterizacao geral da base (observacoes, variaveis, alvo, classes);
  * estatisticas descritivas das variaveis numericas;
  * investigacao da distribuicao das classes;
  * figuras de apoio salvas em ``relatorio/figuras/``;
  * tabelas de apoio salvas em ``resultados/``.

Execucao:
    python src/01_analise_exploratoria.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sem interface grafica
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preparacao import (  # noqa: E402
    ALVO,
    COLUNAS_NUMERICAS_BASE,
    DIR_FIGURAS,
    DIR_RESULTADOS,
    FEATURES,
    GENEROS_SELECIONADOS,
    INFO_FEATURES,
    ROTULOS_PT,
    carregar_base_completa,
    garantir_diretorios,
    preparar_dados,
)

warnings.filterwarnings("ignore")

# Identidade visual das figuras
PALETA = "viridis"
COR_PRINCIPAL = "#1DB954"   # verde Spotify
COR_SECUNDARIA = "#191414"
sns.set_theme(style="whitegrid", palette=PALETA)
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titleweight": "bold",
})


def titulo(texto: str) -> None:
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def salvar(fig: plt.Figure, nome: str) -> None:
    caminho = DIR_FIGURAS / nome
    fig.savefig(caminho)
    plt.close(fig)
    print(f"  [figura] {caminho.relative_to(DIR_FIGURAS.parent.parent)}")


# ---------------------------------------------------------------------------
# 1. Caracterizacao geral da base
# ---------------------------------------------------------------------------

def caracterizar_base(df: pd.DataFrame) -> None:
    titulo("1. CARACTERIZACAO GERAL DA BASE DE DADOS")

    print(f"Fonte .................. Kaggle / maharshipandya - Spotify Tracks Dataset")
    print(f"Quantidade de observacoes .............. {df.shape[0]:,}")
    print(f"Quantidade de colunas .................. {df.shape[1]}")
    print(f"Variavel-alvo .......................... {ALVO}")
    print(f"Classes na variavel-alvo ............... {df[ALVO].nunique()}")
    print(f"Variaveis preditoras candidatas ........ {df.shape[1] - 1 - 4} "
          f"(excluindo o alvo e 4 colunas de identificacao)")

    print("\nTipos de dados por coluna:")
    tipos = (
        df.dtypes.astype(str)
        .rename("tipo")
        .to_frame()
        .assign(
            nulos=df.isna().sum(),
            unicos=df.nunique(),
        )
    )
    print(tipos.to_string())
    tipos.to_csv(DIR_RESULTADOS / "aed_dicionario_colunas.csv")

    print("\nValores ausentes (total):", int(df.isna().sum().sum()))
    print("Faixas (track_id) distintas:", f"{df['track_id'].nunique():,}")
    print("Faixas catalogadas em mais de um genero:",
          f"{(df.groupby('track_id')[ALVO].nunique() > 1).sum():,}")


# ---------------------------------------------------------------------------
# 2. Distribuicao das classes
# ---------------------------------------------------------------------------

def investigar_classes(df_completo: pd.DataFrame, df: pd.DataFrame) -> None:
    titulo("2. INVESTIGACAO DA DISTRIBUICAO DAS CLASSES")

    cont_total = df_completo[ALVO].value_counts()
    print(f"Base completa: {len(cont_total)} generos")
    print(f"  faixas por genero -> min={cont_total.min()}, "
          f"max={cont_total.max()}, media={cont_total.mean():.1f}")
    print("  => a base original e perfeitamente balanceada "
          "(1.000 faixas por genero).")

    cont = df[ALVO].value_counts().sort_index()
    print(f"\nConjunto de modelagem: {len(cont)} generos selecionados")
    tabela = pd.DataFrame({
        "genero": cont.index,
        "rotulo": [ROTULOS_PT[g] for g in cont.index],
        "n_faixas": cont.values,
        "proporcao_%": (100 * cont.values / cont.values.sum()).round(2),
    })
    print(tabela.to_string(index=False))
    tabela.to_csv(DIR_RESULTADOS / "aed_distribuicao_classes.csv", index=False)

    desvio = 100 * cont.std() / cont.mean()
    print(f"\nCoeficiente de variacao entre classes: {desvio:.2f}%")
    print("  => conjunto praticamente balanceado; a acuracia e uma metrica")
    print("     confiavel e a linha de base aleatoria e 1/10 = 10%.")

    # Figura: distribuicao das classes
    fig, eixos = plt.subplots(1, 2, figsize=(14, 5))
    eixos[0].bar(range(len(cont_total)), cont_total.values,
                 color=COR_PRINCIPAL, width=1.0)
    eixos[0].set_title("Base completa: 114 generos x 1.000 faixas")
    eixos[0].set_xlabel("generos (ordenados)")
    eixos[0].set_ylabel("n de faixas")
    eixos[0].set_ylim(0, 1200)

    cores = sns.color_palette(PALETA, len(tabela))
    eixos[1].barh(tabela["rotulo"], tabela["n_faixas"], color=cores)
    eixos[1].set_title("Conjunto de modelagem: 10 classes selecionadas")
    eixos[1].set_xlabel("n de faixas (apos pre-processamento)")
    for i, v in enumerate(tabela["n_faixas"]):
        eixos[1].text(v + 5, i, str(v), va="center", fontsize=9)
    fig.suptitle("Distribuicao da variavel-alvo", fontsize=13, fontweight="bold")
    fig.tight_layout()
    salvar(fig, "01_distribuicao_classes.png")


# ---------------------------------------------------------------------------
# 3. Estatisticas descritivas
# ---------------------------------------------------------------------------

def estatisticas_descritivas(df_completo: pd.DataFrame, df: pd.DataFrame) -> None:
    titulo("3. ESTATISTICAS DESCRITIVAS DAS VARIAVEIS")

    desc = df_completo[COLUNAS_NUMERICAS_BASE].describe().T
    desc["assimetria"] = df_completo[COLUNAS_NUMERICAS_BASE].skew()
    desc["curtose"] = df_completo[COLUNAS_NUMERICAS_BASE].kurtosis()
    desc = desc.round(3)
    print("\n>>> Base completa (14 variaveis numericas):")
    print(desc.to_string())
    desc.to_csv(DIR_RESULTADOS / "aed_estatisticas_base_completa.csv")

    desc_m = df[FEATURES].describe().T
    desc_m["assimetria"] = df[FEATURES].skew()
    desc_m["curtose"] = df[FEATURES].kurtosis()
    desc_m = desc_m.round(3)
    print("\n>>> Conjunto de modelagem (12 features escolhidas):")
    print(desc_m.to_string())
    desc_m.to_csv(DIR_RESULTADOS / "aed_estatisticas_features.csv")

    print("\nLeitura dos coeficientes de assimetria:")
    assimetricas = desc_m["assimetria"].abs().sort_values(ascending=False)
    for nome, valor in assimetricas.head(5).items():
        print(f"  {nome:<18} assimetria = {desc_m.loc[nome, 'assimetria']:+7.2f}")
    print("  => varias features estao longe da normalidade. Esse achado motiva")
    print("     o teste de variantes do Naive Bayes que nao assumem gaussianidade")
    print("     (ver script 02_treinar_modelo.py).")

    # Figura: histogramas das features
    fig, eixos = plt.subplots(3, 4, figsize=(16, 9))
    for ax, col in zip(eixos.ravel(), FEATURES):
        ax.hist(df[col], bins=40, color=COR_PRINCIPAL, edgecolor="white", linewidth=0.3)
        ax.set_title(f"{INFO_FEATURES[col]['rotulo']}\n(assimetria = "
                     f"{df[col].skew():+.2f})", fontsize=9)
        ax.set_yticks([])
    fig.suptitle("Distribuicao das 12 variaveis preditoras (conjunto de modelagem)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    salvar(fig, "02_histogramas_features.png")

    # Figura: boxplots por classe para as features mais discriminantes
    destaques = ["acousticness", "instrumentalness", "speechiness",
                 "energy", "danceability", "loudness"]
    fig, eixos = plt.subplots(2, 3, figsize=(16, 8))
    ordem = sorted(GENEROS_SELECIONADOS)
    for ax, col in zip(eixos.ravel(), destaques):
        sns.boxplot(data=df, x=ALVO, y=col, order=ordem, ax=ax,
                    hue=ALVO, hue_order=ordem, legend=False,
                    palette=PALETA, fliersize=1, linewidth=0.8)
        ax.set_title(INFO_FEATURES[col]["rotulo"], fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=90, labelsize=8)
    fig.suptitle("Separacao das classes pelas features mais discriminantes",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    salvar(fig, "03_boxplots_por_classe.png")


# ---------------------------------------------------------------------------
# 4. Correlacao e poder discriminante (base para a selecao de features)
# ---------------------------------------------------------------------------

def analisar_features(df: pd.DataFrame) -> None:
    titulo("4. CORRELACAO E PODER DISCRIMINANTE DAS FEATURES")

    corr = df[FEATURES].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    mascara = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mascara, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.7, "label": "Spearman"},
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("Correlacao de Spearman entre as variaveis preditoras",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    salvar(fig, "04_matriz_correlacao.png")

    pares = (
        corr.where(mascara).stack()
        .rename("spearman").reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b"})
    )
    pares["abs"] = pares["spearman"].abs()
    fortes = pares.sort_values("abs", ascending=False).head(8)
    print("Pares de features mais correlacionados:")
    print(fortes[["feature_a", "feature_b", "spearman"]].to_string(index=False))
    print("\n  => A correlacao energy/loudness e energy/acousticness viola a")
    print("     hipotese de independencia condicional do Naive Bayes. Mantemos")
    print("     as variaveis por aportarem informacao complementar, e a")
    print("     consequencia (probabilidades super-confiantes) e discutida no")
    print("     relatorio. Este e o 'naive' do algoritmo.")
    fortes.to_csv(DIR_RESULTADOS / "aed_correlacoes_fortes.csv", index=False)

    # Poder discriminante univariado: razao F (ANOVA) de cada feature
    from sklearn.feature_selection import f_classif
    f_val, p_val = f_classif(df[FEATURES], df[ALVO])
    importancia = (
        pd.DataFrame({"feature": FEATURES, "razao_F": f_val, "p_valor": p_val})
        .sort_values("razao_F", ascending=False)
        .reset_index(drop=True)
    )
    print("\nPoder discriminante univariado (ANOVA F entre as 10 classes):")
    print(importancia.to_string(index=False))
    importancia.to_csv(DIR_RESULTADOS / "aed_poder_discriminante.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    cores = sns.color_palette(PALETA, len(importancia))
    ax.barh(
        [INFO_FEATURES[f]["rotulo"] for f in importancia["feature"]],
        importancia["razao_F"], color=cores,
    )
    ax.invert_yaxis()
    ax.set_xlabel("razao F (ANOVA) - escala logaritmica")
    ax.set_xscale("log")
    ax.set_title("Poder discriminante de cada variavel preditora",
                 fontsize=12, fontweight="bold")
    for i, v in enumerate(importancia["razao_F"]):
        ax.text(v * 1.05, i, f"{v:,.0f}", va="center", fontsize=8)
    fig.tight_layout()
    salvar(fig, "05_poder_discriminante.png")

    # Perfil medio por genero (mapa de calor normalizado)
    perfil = df.groupby(ALVO)[FEATURES].mean()
    normalizado = (perfil - perfil.min()) / (perfil.max() - perfil.min())
    normalizado.index = [ROTULOS_PT[g] for g in normalizado.index]
    normalizado.columns = [INFO_FEATURES[c]["rotulo"] for c in normalizado.columns]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(normalizado, annot=perfil.round(2).values, fmt="g",
                cmap=PALETA, linewidths=0.5, annot_kws={"size": 7},
                cbar_kws={"label": "valor normalizado por coluna"}, ax=ax)
    ax.set_title("Perfil acustico medio de cada genero\n"
                 "(cor = posicao relativa; numero = media real)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    salvar(fig, "06_perfil_medio_generos.png")
    perfil.round(4).to_csv(DIR_RESULTADOS / "aed_perfil_medio_generos.csv")

    print("\nPerfil medio por genero salvo em resultados/aed_perfil_medio_generos.csv")


def main() -> None:
    garantir_diretorios()

    print("\n" + "#" * 78)
    print("#  ANALISE EXPLORATORIA - CLASSIFICACAO DE GENEROS MUSICAIS")
    print("#  Trabalho Final de Probabilidade e Estatistica - IFCE")
    print("#" * 78)

    df_completo = carregar_base_completa()
    caracterizar_base(df_completo)

    titulo("PRE-PROCESSAMENTO")
    df = preparar_dados(df_completo)

    investigar_classes(df_completo, df)
    estatisticas_descritivas(df_completo, df)
    analisar_features(df)

    titulo("ANALISE EXPLORATORIA CONCLUIDA")
    print(f"Figuras geradas em ...... {DIR_FIGURAS}")
    print(f"Tabelas geradas em ...... {DIR_RESULTADOS}")
    print("\nProximo passo: python src/02_treinar_modelo.py")


if __name__ == "__main__":
    main()
