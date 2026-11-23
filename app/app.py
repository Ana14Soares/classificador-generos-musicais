"""
app.py
======
Back-end da aplicacao web - Flask.

Responsabilidades (conforme o enunciado do trabalho):
  * Carregar o modelo Naive Bayes salvo em disco (joblib);
  * Receber os dados enviados pelo formulario;
  * Aplicar o modelo;
  * Retornar o resultado (classe prevista e probabilidades) ao usuario.

Execucao:
    python app/app.py
    -> http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preparacao import (  # noqa: E402
    CAMINHO_METADADOS,
    CAMINHO_MODELO,
    EMOJIS,
    FEATURES,
    INFO_FEATURES,
    ROTULOS_PT,
)

app = Flask(__name__)

CAMINHO_EXEMPLOS = CAMINHO_MODELO.parent / "exemplos.json"


# ---------------------------------------------------------------------------
# Carregamento dos artefatos (uma unica vez, na inicializacao do servidor)
# ---------------------------------------------------------------------------

def carregar_artefatos():
    """Carrega modelo, metadados e exemplos; encerra com mensagem clara se faltarem."""
    if not CAMINHO_MODELO.exists():
        raise SystemExit(
            "\n[ERRO] Modelo nao encontrado em "
            f"{CAMINHO_MODELO}\n"
            "Treine o modelo antes de iniciar a aplicacao:\n"
            "    python src/02_treinar_modelo.py\n"
        )

    modelo = joblib.load(CAMINHO_MODELO)
    metadados = json.loads(CAMINHO_METADADOS.read_text(encoding="utf-8"))

    exemplos = []
    if CAMINHO_EXEMPLOS.exists():
        exemplos = json.loads(CAMINHO_EXEMPLOS.read_text(encoding="utf-8"))

    return modelo, metadados, exemplos


MODELO, METADADOS, EXEMPLOS = carregar_artefatos()
CLASSES = list(MODELO.classes_)

print("=" * 70)
print("  Classificador de Generos Musicais - Naive Bayes")
print("=" * 70)
print(f"  Modelo ........ {METADADOS['algoritmo']}")
print(f"  Treinado em ... {METADADOS['treinado_em']}")
print(f"  Acuracia ...... {METADADOS['metricas_teste']['acuracia']:.2%}")
print(f"  Classes ....... {len(CLASSES)}")
print(f"  Exemplos ...... {len(EXEMPLOS)} faixas reais disponiveis")
print("=" * 70)


# ---------------------------------------------------------------------------
# Validacao da entrada
# ---------------------------------------------------------------------------

def extrair_features(dados: dict) -> tuple[pd.DataFrame | None, str | None]:
    """Converte o payload do formulario em um DataFrame de uma linha.

    Retorna ``(dataframe, None)`` em caso de sucesso ou ``(None, mensagem)``
    quando a entrada e invalida.
    """
    valores: dict[str, float] = {}

    for feature in FEATURES:
        bruto = dados.get(feature)
        if bruto is None or str(bruto).strip() == "":
            return None, f"O campo '{INFO_FEATURES[feature]['rotulo']}' é obrigatório."

        try:
            valor = float(bruto)
        except (TypeError, ValueError):
            return None, (f"O campo '{INFO_FEATURES[feature]['rotulo']}' deve ser "
                          "um número válido.")

        info = INFO_FEATURES[feature]
        if not (info["min"] <= valor <= info["max"]):
            return None, (f"O campo '{info['rotulo']}' deve estar entre "
                          f"{info['min']:g} e {info['max']:g}.")

        valores[feature] = valor

    return pd.DataFrame([valores], columns=FEATURES), None


# ---------------------------------------------------------------------------
# Rotas
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Pagina principal com o formulario de entrada."""
    return render_template(
        "index.html",
        features=FEATURES,
        info=INFO_FEATURES,
        metadados=METADADOS,
        rotulos=ROTULOS_PT,
        emojis=EMOJIS,
        classes=CLASSES,
    )


@app.route("/prever", methods=["POST"])
def prever():
    """Recebe os dados do formulario, aplica o modelo e devolve a previsao."""
    dados = request.get_json(silent=True) or request.form.to_dict()

    X, erro = extrair_features(dados)
    if erro is not None:
        return jsonify({"ok": False, "erro": erro}), 400

    classe = MODELO.predict(X)[0]
    probabilidades = MODELO.predict_proba(X)[0]

    ranking = sorted(
        (
            {
                "classe": c,
                "rotulo": ROTULOS_PT.get(c, c),
                "emoji": EMOJIS.get(c, ""),
                "probabilidade": float(p),
                "percentual": round(float(p) * 100, 2),
            }
            for c, p in zip(CLASSES, probabilidades)
        ),
        key=lambda d: d["probabilidade"],
        reverse=True,
    )

    confianca = ranking[0]["probabilidade"]
    if confianca >= 0.90:
        nivel, texto_nivel = "alta", "O modelo está bastante seguro desta previsão."
    elif confianca >= 0.60:
        nivel, texto_nivel = "media", "O modelo tem confiança moderada nesta previsão."
    else:
        nivel, texto_nivel = "baixa", ("A faixa fica na fronteira entre gêneros; "
                                       "veja as alternativas abaixo.")

    return jsonify({
        "ok": True,
        "classe": classe,
        "rotulo": ROTULOS_PT.get(classe, classe),
        "emoji": EMOJIS.get(classe, ""),
        "confianca": round(confianca * 100, 2),
        "nivel_confianca": nivel,
        "texto_confianca": texto_nivel,
        "ranking": ranking,
    })


@app.route("/exemplo")
def exemplo():
    """Devolve uma faixa real da base, para preencher o formulario.

    Util na apresentacao: permite comparar a previsao do modelo com o
    genero verdadeiro da faixa.
    """
    if not EXEMPLOS:
        return jsonify({"ok": False, "erro": "Nenhum exemplo disponível."}), 404

    genero = request.args.get("genero")
    candidatos = [e for e in EXEMPLOS if e["track_genre"] == genero] if genero else EXEMPLOS
    if not candidatos:
        candidatos = EXEMPLOS

    faixa = random.choice(candidatos)
    return jsonify({
        "ok": True,
        "nome": faixa.get("track_name", "(sem título)"),
        "artista": faixa.get("artists", "(desconhecido)"),
        "genero_real": faixa["track_genre"],
        "rotulo_real": ROTULOS_PT.get(faixa["track_genre"], faixa["track_genre"]),
        "valores": {f: faixa[f] for f in FEATURES},
    })


@app.route("/sobre")
def sobre():
    """Expoe os metadados do modelo (usado pelo painel 'Sobre o modelo')."""
    return jsonify(METADADOS)


@app.errorhandler(404)
def nao_encontrado(_):
    return jsonify({"ok": False, "erro": "Rota não encontrada."}), 404


@app.errorhandler(500)
def erro_interno(_):
    return jsonify({"ok": False, "erro": "Erro interno no servidor."}), 500


if __name__ == "__main__":
    import os

    # A porta pode ser alterada via variavel de ambiente PORT. No macOS a porta
    # 5000 costuma estar ocupada pelo servico "AirPlay Receiver"; nesse caso use:
    #     PORT=5050 python app/app.py
    porta = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="127.0.0.1", port=porta)
