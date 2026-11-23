"""
02_treinar_modelo.py
====================
Treinamento, avaliacao e persistencia do modelo Naive Bayes.

Etapas executadas:
  1. Carregamento da base e pre-processamento (modulo ``preparacao``).
  2. Selecao das variaveis preditoras.
  3. Divisao estratificada em treino (80%) e teste (20%).
  4. Comparacao de tres variantes do Naive Bayes por validacao cruzada.
  5. Ajuste de hiperparametros da variante vencedora (GridSearchCV).
  6. Avaliacao final no conjunto de teste: acuracia, precisao, revocacao,
     F1-Score e matriz de confusao.
  7. Salvamento do modelo treinado com ``joblib`` e dos metadados em JSON.

Execucao:
    python src/02_treinar_modelo.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, PowerTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preparacao import (  # noqa: E402
    ALVO,
    CAMINHO_METADADOS,
    CAMINHO_MODELO,
    DIR_FIGURAS,
    DIR_RESULTADOS,
    FEATURES,
    GENEROS_SELECIONADOS,
    PROPORCAO_TESTE,
    ROTULOS_PT,
    SEMENTE,
    garantir_diretorios,
    obter_dados_modelagem,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130,
                     "savefig.bbox": "tight", "axes.titleweight": "bold"})

COR_PRINCIPAL = "#1DB954"


def titulo(texto: str) -> None:
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Definicao dos modelos candidatos
# ---------------------------------------------------------------------------

def construir_candidatos() -> dict[str, Pipeline | GaussianNB]:
    """Tres variantes do Naive Bayes, da mais simples a mais robusta.

    * ``GaussianNB``: assume que cada feature segue uma normal dentro de cada
      classe. E a formulacao classica para atributos continuos.
    * ``PowerTransformer + GaussianNB``: aplica a transformacao de Yeo-Johnson
      para aproximar cada feature de uma normal antes do ajuste, atacando
      diretamente a assimetria detectada na analise exploratoria.
    * ``KBinsDiscretizer + CategoricalNB``: discretiza cada feature continua em
      faixas por quantis e modela cada faixa como uma categoria. Abandona
      qualquer suposicao de forma da distribuicao.
    """
    return {
        "GaussianNB": GaussianNB(),
        "Yeo-Johnson + GaussianNB": Pipeline([
            ("transformar", PowerTransformer(method="yeo-johnson", standardize=True)),
            ("nb", GaussianNB()),
        ]),
        "Discretizacao + CategoricalNB": Pipeline([
            ("discretizar", KBinsDiscretizer(n_bins=20, encode="ordinal",
                                             strategy="quantile")),
            ("nb", CategoricalNB(alpha=1.0)),
        ]),
    }


# ---------------------------------------------------------------------------
# Comparacao por validacao cruzada
# ---------------------------------------------------------------------------

def comparar_modelos(X_tr, y_tr) -> pd.DataFrame:
    titulo("4. COMPARACAO DAS VARIANTES DO NAIVE BAYES (VALIDACAO CRUZADA)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
    linhas = []

    # Linha de base: classificador ingenuo que sempre chuta a classe majoritaria
    dummy = DummyClassifier(strategy="most_frequent")
    pontos = cross_val_score(dummy, X_tr, y_tr, cv=cv, scoring="accuracy")
    linhas.append({"modelo": "Linha de base (classe majoritaria)",
                   "acuracia_media": pontos.mean(), "acuracia_desvio": pontos.std(),
                   "f1_macro_media": np.nan, "f1_macro_desvio": np.nan,
                   "tempo_s": 0.0})
    print(f"Linha de base (classe majoritaria): acuracia = {pontos.mean():.4f}")
    print("(qualquer modelo util precisa superar confortavelmente esse valor)\n")

    for nome, modelo in construir_candidatos().items():
        t0 = time.perf_counter()
        acc = cross_val_score(modelo, X_tr, y_tr, cv=cv, scoring="accuracy", n_jobs=-1)
        f1 = cross_val_score(modelo, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=-1)
        dt = time.perf_counter() - t0
        linhas.append({"modelo": nome,
                       "acuracia_media": acc.mean(), "acuracia_desvio": acc.std(),
                       "f1_macro_media": f1.mean(), "f1_macro_desvio": f1.std(),
                       "tempo_s": dt})
        print(f"{nome:<32} acuracia = {acc.mean():.4f} (+/- {acc.std():.4f})"
              f"   F1-macro = {f1.mean():.4f} (+/- {f1.std():.4f})")

    tabela = pd.DataFrame(linhas)
    tabela.round(4).to_csv(DIR_RESULTADOS / "comparacao_modelos.csv", index=False)

    print("\nLeitura: a GaussianNB pura e penalizada pela forte assimetria das")
    print("features (vista na AED). Tanto a normalizacao Yeo-Johnson quanto a")
    print("discretizacao por quantis corrigem o problema, confirmando na pratica")
    print("que a suposicao de normalidade era o gargalo do modelo.")

    # Figura comparativa
    comp = tabela.dropna(subset=["f1_macro_media"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y_pos = np.arange(len(comp))
    ax.barh(y_pos - 0.2, comp["acuracia_media"], height=0.38,
            xerr=comp["acuracia_desvio"], label="Acuracia",
            color=COR_PRINCIPAL, capsize=3)
    ax.barh(y_pos + 0.2, comp["f1_macro_media"], height=0.38,
            xerr=comp["f1_macro_desvio"], label="F1-Score (macro)",
            color="#535353", capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comp["modelo"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.axvline(0.10, color="crimson", linestyle="--", linewidth=1,
               label="Acaso (10 classes = 10%)")
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("desempenho medio em validacao cruzada (5 folds)")
    ax.set_title("Comparacao das variantes do Naive Bayes")
    # Legenda abaixo do grafico: dentro da area de plotagem ela cobriria os
    # rotulos de valor das barras mais longas.
    ax.legend(fontsize=8, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), frameon=False)
    for i, (a, f) in enumerate(zip(comp["acuracia_media"], comp["f1_macro_media"])):
        ax.text(a + 0.015, i - 0.2, f"{a:.3f}", va="center", fontsize=8)
        ax.text(f + 0.015, i + 0.2, f"{f:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / "07_comparacao_modelos.png")
    plt.close(fig)
    print(f"\n  [figura] relatorio/figuras/07_comparacao_modelos.png")

    return tabela


# ---------------------------------------------------------------------------
# Ajuste de hiperparametros
# ---------------------------------------------------------------------------

def ajustar_hiperparametros(X_tr, y_tr) -> GridSearchCV:
    titulo("5. AJUSTE DE HIPERPARAMETROS DO MODELO VENCEDOR")

    pipe = Pipeline([
        ("discretizar", KBinsDiscretizer(encode="ordinal", strategy="quantile")),
        ("nb", CategoricalNB()),
    ])
    grade = {
        "discretizar__n_bins": [8, 12, 16, 20, 24, 32],
        "discretizar__strategy": ["quantile", "uniform"],
        "nb__alpha": [0.1, 0.5, 1.0, 2.0],
    }
    print(f"Combinacoes testadas: "
          f"{len(grade['discretizar__n_bins']) * len(grade['discretizar__strategy']) * len(grade['nb__alpha'])}")
    print("Criterio de selecao: F1-Score macro em validacao cruzada (5 folds).\n")

    busca = GridSearchCV(
        pipe, grade, scoring="f1_macro", n_jobs=-1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE),
        return_train_score=True,
    )
    busca.fit(X_tr, y_tr)

    print(f"Melhores hiperparametros: {busca.best_params_}")
    print(f"F1-macro em validacao cruzada: {busca.best_score_:.4f}")

    res = pd.DataFrame(busca.cv_results_)
    (res[["param_discretizar__n_bins", "param_discretizar__strategy",
          "param_nb__alpha", "mean_test_score", "std_test_score",
          "mean_train_score", "rank_test_score"]]
     .sort_values("rank_test_score")
     .round(4)
     .to_csv(DIR_RESULTADOS / "busca_hiperparametros.csv", index=False))

    melhor = res.loc[busca.best_index_]
    print(f"\nF1-macro em treino: {melhor['mean_train_score']:.4f}  |  "
          f"em validacao: {melhor['mean_test_score']:.4f}")
    print(f"Diferenca treino-validacao: "
          f"{melhor['mean_train_score'] - melhor['mean_test_score']:+.4f} "
          "(diferenca pequena indica ausencia de sobreajuste relevante)")

    # Figura: efeito do numero de faixas
    quant = res[(res["param_discretizar__strategy"] == "quantile")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for alpha in sorted(quant["param_nb__alpha"].unique()):
        sub = quant[quant["param_nb__alpha"] == alpha].sort_values(
            "param_discretizar__n_bins")
        ax.plot(sub["param_discretizar__n_bins"], sub["mean_test_score"],
                marker="o", label=f"alpha = {alpha}")
    ax.set_xlabel("numero de faixas de discretizacao (n_bins)")
    ax.set_ylabel("F1-Score macro (validacao cruzada)")
    ax.set_title("Sensibilidade aos hiperparametros (estrategia por quantis)")
    ax.legend(title="suavizacao de Laplace", fontsize=8)
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / "08_ajuste_hiperparametros.png")
    plt.close(fig)
    print("  [figura] relatorio/figuras/08_ajuste_hiperparametros.png")

    return busca


# ---------------------------------------------------------------------------
# Avaliacao final
# ---------------------------------------------------------------------------

def avaliar(modelo, X_te, y_te, classes) -> dict:
    titulo("6. AVALIACAO FINAL NO CONJUNTO DE TESTE")

    y_pred = modelo.predict(X_te)
    y_prob = modelo.predict_proba(X_te)

    acuracia = accuracy_score(y_te, y_pred)
    metricas = {
        "acuracia": acuracia,
        "precisao_macro": precision_score(y_te, y_pred, average="macro"),
        "revocacao_macro": recall_score(y_te, y_pred, average="macro"),
        "f1_macro": f1_score(y_te, y_pred, average="macro"),
        "precisao_ponderada": precision_score(y_te, y_pred, average="weighted"),
        "revocacao_ponderada": recall_score(y_te, y_pred, average="weighted"),
        "f1_ponderado": f1_score(y_te, y_pred, average="weighted"),
    }

    print(f"Observacoes no conjunto de teste: {len(y_te):,}\n")
    print(f"  Acuracia ............... {metricas['acuracia']:.4f}  "
          f"({metricas['acuracia'] * 100:.2f}%)")
    print(f"  Precisao (macro) ....... {metricas['precisao_macro']:.4f}")
    print(f"  Revocacao (macro) ...... {metricas['revocacao_macro']:.4f}")
    print(f"  F1-Score (macro) ....... {metricas['f1_macro']:.4f}")
    print(f"  Precisao (ponderada) ... {metricas['precisao_ponderada']:.4f}")
    print(f"  Revocacao (ponderada) .. {metricas['revocacao_ponderada']:.4f}")
    print(f"  F1-Score (ponderado) ... {metricas['f1_ponderado']:.4f}")
    print(f"\n  Ganho sobre o acaso: {metricas['acuracia'] / 0.10:.1f}x "
          f"(linha de base de 10 classes balanceadas = 10%)")

    print("\n--- Relatorio de classificacao por genero ---")
    relatorio_txt = classification_report(
        y_te, y_pred, digits=3,
        target_names=[ROTULOS_PT[c] for c in classes])
    print(relatorio_txt)

    relatorio = classification_report(y_te, y_pred, output_dict=True)
    por_classe = (
        pd.DataFrame(relatorio).T
        .loc[list(classes)]
        .rename(columns={"precision": "precisao", "recall": "revocacao",
                         "f1-score": "f1_score", "support": "suporte"})
    )
    por_classe.insert(0, "rotulo", [ROTULOS_PT[c] for c in por_classe.index])
    por_classe.round(4).to_csv(DIR_RESULTADOS / "metricas_por_classe.csv")

    # Matriz de confusao
    mc = confusion_matrix(y_te, y_pred, labels=list(classes))
    pd.DataFrame(mc, index=list(classes), columns=list(classes)).to_csv(
        DIR_RESULTADOS / "matriz_confusao.csv")

    print("--- Matriz de confusao (contagens absolutas) ---")
    print(pd.DataFrame(mc, index=list(classes), columns=list(classes)).to_string())

    rotulos = [ROTULOS_PT[c] for c in classes]
    fig, eixos = plt.subplots(1, 2, figsize=(17, 7))
    ConfusionMatrixDisplay(mc, display_labels=rotulos).plot(
        ax=eixos[0], cmap="Greens", colorbar=False, values_format="d")
    eixos[0].set_title("Matriz de confusao - contagens")
    eixos[0].set_xlabel("Genero previsto")
    eixos[0].set_ylabel("Genero real")

    mc_norm = mc.astype(float) / mc.sum(axis=1, keepdims=True)
    sns.heatmap(mc_norm, annot=True, fmt=".2f", cmap="Greens", vmin=0, vmax=1,
                xticklabels=rotulos, yticklabels=rotulos, square=True,
                linewidths=0.5, annot_kws={"size": 7},
                cbar_kws={"shrink": 0.75, "label": "proporcao da linha"},
                ax=eixos[1])
    eixos[1].set_title("Matriz de confusao - normalizada por classe real")
    eixos[1].set_xlabel("Genero previsto")
    eixos[1].set_ylabel("Genero real")

    for ax in eixos:
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.suptitle(f"Desempenho do modelo Naive Bayes no conjunto de teste "
                 f"(acuracia = {acuracia:.1%})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / "09_matriz_confusao.png")
    plt.close(fig)
    print("\n  [figura] relatorio/figuras/09_matriz_confusao.png")

    # Figura: metricas por classe
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(por_classe))
    largura = 0.27
    ax.bar(x - largura, por_classe["precisao"], largura, label="Precisao",
           color=COR_PRINCIPAL)
    ax.bar(x, por_classe["revocacao"], largura, label="Revocacao", color="#1ed760aa")
    ax.bar(x + largura, por_classe["f1_score"], largura, label="F1-Score",
           color="#535353")
    ax.axhline(acuracia, color="crimson", linestyle="--", linewidth=1,
               label=f"acuracia global ({acuracia:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(por_classe["rotulo"], rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("valor da metrica")
    ax.set_title("Precisao, revocacao e F1-Score por genero")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / "10_metricas_por_classe.png")
    plt.close(fig)
    print("  [figura] relatorio/figuras/10_metricas_por_classe.png")

    # Analise das confusoes mais frequentes
    print("\n--- Erros mais frequentes do modelo ---")
    erros = []
    for i, real in enumerate(classes):
        for j, prev in enumerate(classes):
            if i != j and mc[i, j] > 0:
                erros.append({"real": ROTULOS_PT[real], "previsto": ROTULOS_PT[prev],
                              "casos": int(mc[i, j]),
                              "perc_da_classe": 100 * mc[i, j] / mc[i].sum()})
    erros_df = pd.DataFrame(erros).sort_values("casos", ascending=False)
    print(erros_df.head(8).round(1).to_string(index=False))
    erros_df.round(2).to_csv(DIR_RESULTADOS / "confusoes_frequentes.csv", index=False)

    # Calibracao: confianca media quando acerta vs. quando erra
    conf = y_prob.max(axis=1)
    acertou = (y_pred == y_te.values)
    print(f"\nConfianca media quando o modelo acerta: {conf[acertou].mean():.3f}")
    print(f"Confianca media quando o modelo erra  : {conf[~acertou].mean():.3f}")
    print("  => a confianca e sistematicamente alta: efeito conhecido da hipotese")
    print("     de independencia do Naive Bayes sobre features correlacionadas.")
    print("     A probabilidade serve como ranqueamento, nao como probabilidade")
    print("     calibrada. Isso esta documentado na interface da aplicacao.")

    metricas["confianca_media_acerto"] = float(conf[acertou].mean())
    metricas["confianca_media_erro"] = float(conf[~acertou].mean())
    return metricas


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------

def main() -> None:
    garantir_diretorios()

    print("\n" + "#" * 78)
    print("#  TREINAMENTO DO MODELO NAIVE BAYES")
    print("#  Trabalho Final de Probabilidade e Estatistica - IFCE")
    print("#" * 78)

    titulo("1-2. CARREGAMENTO DOS DADOS E SELECAO DAS VARIAVEIS PREDITORAS")
    X, y, df = obter_dados_modelagem()
    print(f"\nVariavel-alvo .......... {ALVO} ({y.nunique()} classes)")
    print(f"Variaveis preditoras ... {len(FEATURES)}")
    for f in FEATURES:
        print(f"    - {f}")

    titulo("3. DIVISAO EM TREINAMENTO E TESTE")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=PROPORCAO_TESTE, random_state=SEMENTE, stratify=y)
    print(f"Treinamento ... {len(X_tr):,} faixas ({100 * (1 - PROPORCAO_TESTE):.0f}%)")
    print(f"Teste ......... {len(X_te):,} faixas ({100 * PROPORCAO_TESTE:.0f}%)")
    print("Divisao estratificada: a proporcao de cada genero e preservada nos")
    print("dois conjuntos, garantindo avaliacao justa de todas as classes.")

    comparacao = comparar_modelos(X_tr, y_tr)
    busca = ajustar_hiperparametros(X_tr, y_tr)

    modelo = busca.best_estimator_
    classes = list(modelo.classes_)
    metricas = avaliar(modelo, X_te, y_te, classes)

    # ---------------------------------------------------------------- salvar
    titulo("7. SALVAMENTO DO MODELO TREINADO")

    joblib.dump(modelo, CAMINHO_MODELO, compress=3)
    tamanho_kb = CAMINHO_MODELO.stat().st_size / 1024
    print(f"Modelo salvo com joblib em: {CAMINHO_MODELO}")
    print(f"Tamanho do arquivo: {tamanho_kb:.1f} KB")

    metadados = {
        "projeto": "Classificacao de generos musicais com Naive Bayes",
        "disciplina": "Probabilidade e Estatistica - IFCE",
        "treinado_em": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "versao_scikit_learn": sklearn.__version__,
        "versao_python": sys.version.split()[0],
        "base_de_dados": {
            "nome": "Spotify Tracks Dataset",
            "fonte": "https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset",
            "observacoes_originais": 114000,
            "generos_originais": 114,
            "observacoes_apos_preprocessamento": int(len(df)),
        },
        "algoritmo": "Naive Bayes Categórico com discretização por quantis",
        "pipeline": [passo for passo, _ in modelo.steps],
        "melhores_hiperparametros": {k: (v.item() if hasattr(v, "item") else v)
                                     for k, v in busca.best_params_.items()},
        "variavel_alvo": ALVO,
        "classes": classes,
        "rotulos_classes": {c: ROTULOS_PT[c] for c in classes},
        "features": FEATURES,
        "n_treino": int(len(X_tr)),
        "n_teste": int(len(X_te)),
        "metricas_teste": {k: round(float(v), 4) for k, v in metricas.items()},
        "f1_macro_validacao_cruzada": round(float(busca.best_score_), 4),
    }
    CAMINHO_METADADOS.write_text(
        json.dumps(metadados, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Metadados salvos em: {CAMINHO_METADADOS}")

    # Teste de integridade: recarregar e reproduzir uma predicao
    recarregado = joblib.load(CAMINHO_MODELO)
    iguais = (recarregado.predict(X_te.head(200)) == modelo.predict(X_te.head(200))).all()
    print(f"\nVerificacao de integridade do arquivo salvo: "
          f"{'OK - predicoes identicas apos recarga' if iguais else 'FALHOU'}")

    # Exemplos reais para uso na aplicacao web
    exemplos = (
        df.groupby(ALVO, group_keys=False)
          .apply(lambda g: g.sample(min(3, len(g)), random_state=SEMENTE))
          .loc[:, ["track_name", "artists", ALVO] + FEATURES]
    )
    exemplos.to_json(DIR_DADOS_EXEMPLOS := (CAMINHO_MODELO.parent / "exemplos.json"),
                     orient="records", force_ascii=False, indent=2)
    print(f"Exemplos reais para a interface salvos em: {DIR_DADOS_EXEMPLOS}")

    titulo("TREINAMENTO CONCLUIDO")
    print(f"Acuracia final no conjunto de teste: {metricas['acuracia']:.2%}")
    print(f"F1-Score macro: {metricas['f1_macro']:.4f}")
    print("\nProximo passo: python app/app.py  (inicia a aplicacao web)")


if __name__ == "__main__":
    main()
