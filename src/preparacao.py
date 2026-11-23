"""
preparacao.py
=============
Modulo central de configuracao e preparacao dos dados do projeto.

Trabalho Final - Probabilidade e Estatistica (IFCE)
Classificacao supervisionada de generos musicais com Naive Bayes.

Este modulo concentra:
  * os caminhos do projeto;
  * a definicao da variavel-alvo e das features;
  * a rotina de limpeza / pre-processamento reutilizada pela analise
    exploratoria, pelo treinamento e pela aplicacao web.

Manter essa logica em um unico lugar garante que o pre-processamento
aplicado no treinamento seja exatamente o mesmo aplicado na predicao.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Caminhos do projeto
# ---------------------------------------------------------------------------

RAIZ = Path(__file__).resolve().parent.parent

CAMINHO_DADOS_BRUTOS = RAIZ / "dados" / "raw" / "dataset.csv"
DIR_DADOS = RAIZ / "dados"
DIR_MODELO = RAIZ / "modelo"
DIR_FIGURAS = RAIZ / "relatorio" / "figuras"
DIR_RESULTADOS = RAIZ / "resultados"

CAMINHO_MODELO = DIR_MODELO / "modelo_naive_bayes.joblib"
CAMINHO_METADADOS = DIR_MODELO / "metadados_modelo.json"

# ---------------------------------------------------------------------------
# Definicao do problema de classificacao
# ---------------------------------------------------------------------------

#: Nome da coluna usada como variavel-alvo (categorica).
ALVO = "track_genre"

#: Generos selecionados para o problema de classificacao.
#
#  Justificativa (detalhada no relatorio): a base original possui 114 generos
#  com forte sobreposicao semantica ("rock", "hard-rock", "punk-rock",
#  "alt-rock", ...). Essa granularidade torna as classes acusticamente
#  indistinguiveis e derruba o desempenho de qualquer classificador.
#  Selecionamos 10 generos musicalmente contrastantes, mantendo o problema
#  multiclasse, balanceado e com fronteiras interpretaveis.
GENEROS_SELECIONADOS = [
    "classical",
    "comedy",
    "death-metal",
    "forro",
    "hip-hop",
    "jazz",
    "opera",
    "reggaeton",
    "sleep",
    "techno",
]

#: Rotulos amigaveis (exibidos na aplicacao web).
ROTULOS_PT = {
    "classical": "Música Clássica",
    "comedy": "Comédia / Falado",
    "death-metal": "Death Metal",
    "forro": "Forró",
    "hip-hop": "Hip-Hop",
    "jazz": "Jazz",
    "opera": "Ópera",
    "reggaeton": "Reggaeton",
    "sleep": "Sleep / Relaxamento",
    "techno": "Techno",
}

#: Emoji ilustrativo por classe (usado apenas na interface).
EMOJIS = {
    "classical": "\U0001F3BB",
    "comedy": "\U0001F3A4",
    "death-metal": "\U0001F3B8",
    "forro": "\U0001FA97",
    "hip-hop": "\U0001F3A7",
    "jazz": "\U0001F3B7",
    "opera": "\U0001F3AD",
    "reggaeton": "\U0001F483",
    "sleep": "\U0001F634",
    "techno": "\U0001F39B️",
}

#: Variaveis preditoras (features) utilizadas no treinamento.
FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "explicit",
    "popularity",
]

#: Metadados de cada feature para documentacao e para a construcao do
#: formulario da aplicacao web (minimo, maximo, passo, unidade, descricao).
INFO_FEATURES = {
    "danceability": {
        "rotulo": "Dançabilidade",
        "min": 0.0, "max": 1.0, "passo": 0.001, "padrao": 0.5,
        "descricao": "Quão adequada a faixa é para dançar (0 = nada, 1 = muito).",
    },
    "energy": {
        "rotulo": "Energia",
        "min": 0.0, "max": 1.0, "passo": 0.001, "padrao": 0.5,
        "descricao": "Intensidade e atividade percebidas na faixa.",
    },
    "loudness": {
        "rotulo": "Volume médio (dB)",
        "min": -60.0, "max": 5.0, "passo": 0.1, "padrao": -10.0,
        "descricao": "Volume global da faixa em decibéis.",
    },
    "speechiness": {
        "rotulo": "Presença de fala",
        "min": 0.0, "max": 1.0, "passo": 0.001, "padrao": 0.08,
        "descricao": "Proporção de palavras faladas na gravação.",
    },
    "acousticness": {
        "rotulo": "Acusticidade",
        "min": 0.0, "max": 1.0, "passo": 0.001, "padrao": 0.3,
        "descricao": "Confiança de que a faixa é acústica.",
    },
    "instrumentalness": {
        "rotulo": "Instrumentalidade",
        "min": 0.0, "max": 1.0, "passo": 0.001, "padrao": 0.1,
        "descricao": "Confiança de que a faixa não possui vocais.",
    },
    "liveness": {
        "rotulo": "Presença de plateia",
        "min": 0.0, "max": 1.0, "passo": 0.001, "padrao": 0.2,
        "descricao": "Indício de que a faixa foi gravada ao vivo.",
    },
    "valence": {
        "rotulo": "Valência (positividade)",
        "min": 0.0, "max": 1.0, "passo": 0.001, "padrao": 0.5,
        "descricao": "Positividade musical transmitida (0 = triste, 1 = alegre).",
    },
    "tempo": {
        "rotulo": "Tempo (BPM)",
        "min": 0.0, "max": 250.0, "passo": 0.1, "padrao": 120.0,
        "descricao": "Andamento estimado em batidas por minuto.",
    },
    "duration_ms": {
        "rotulo": "Duração (ms)",
        "min": 5000.0, "max": 900000.0, "passo": 1000.0, "padrao": 210000.0,
        "descricao": "Duração total da faixa em milissegundos.",
    },
    "explicit": {
        "rotulo": "Conteúdo explícito",
        "min": 0.0, "max": 1.0, "passo": 1.0, "padrao": 0.0,
        "descricao": "1 se a faixa possui letra explícita, 0 caso contrário.",
    },
    "popularity": {
        "rotulo": "Popularidade",
        "min": 0.0, "max": 100.0, "passo": 1.0, "padrao": 40.0,
        "descricao": "Índice de popularidade da faixa no Spotify (0 a 100).",
    },
}

#: Colunas descritivas (nao usadas como preditoras, mas uteis na analise).
COLUNAS_IDENTIFICACAO = ["track_id", "artists", "album_name", "track_name"]

#: Colunas numericas originais da base (para as estatisticas descritivas).
COLUNAS_NUMERICAS_BASE = [
    "popularity", "duration_ms", "danceability", "energy", "key", "loudness",
    "mode", "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "time_signature",
]

SEMENTE = 42
PROPORCAO_TESTE = 0.20


# ---------------------------------------------------------------------------
# Carregamento e preparacao
# ---------------------------------------------------------------------------

def carregar_base_completa(caminho: Path | None = None) -> pd.DataFrame:
    """Carrega a base bruta completa (114.000 faixas, 114 generos).

    A primeira coluna do CSV original e um indice sem nome, descartado aqui.
    """
    caminho = caminho or CAMINHO_DADOS_BRUTOS
    if not caminho.exists():
        raise FileNotFoundError(
            f"Base de dados nao encontrada em {caminho}.\n"
            "Baixe o arquivo 'dataset.csv' conforme instrucoes do README."
        )
    return pd.read_csv(caminho, index_col=0)


def preparar_dados(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Aplica o pre-processamento completo e retorna o conjunto modelavel.

    Etapas (todas justificadas no relatorio):

    1. Remocao de registros com valores ausentes nas colunas descritivas.
    2. Remocao de duplicatas exatas (mesma faixa repetida no mesmo genero).
    3. Filtragem dos 10 generos selecionados.
    4. Remocao de faixas ambiguas: a mesma ``track_id`` catalogada em mais de
       um dos generos selecionados. Sao rotulos contraditorios para o mesmo
       ponto no espaco de atributos e precisam sair, caso contrario o modelo
       e avaliado contra um rotulo impossivel de acertar.
    5. Conversao de ``explicit`` (booleano) para inteiro 0/1.
    6. Descarte de faixas com ``tempo`` ou ``duration_ms`` iguais a zero
       (registros invalidos na base original).
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    n0 = len(df)
    log(f"Registros brutos ......................... {n0:>7,}")

    df = df.dropna(subset=COLUNAS_IDENTIFICACAO).copy()
    log(f"Apos remover valores ausentes ............ {len(df):>7,}")

    df = df.drop_duplicates(subset=["track_id", ALVO])
    log(f"Apos remover duplicatas exatas ........... {len(df):>7,}")

    df = df[df[ALVO].isin(GENEROS_SELECIONADOS)]
    log(f"Apos filtrar os 10 generos ............... {len(df):>7,}")

    generos_por_faixa = df.groupby("track_id")[ALVO].nunique()
    faixas_unicas = generos_por_faixa[generos_por_faixa == 1].index
    df = df[df["track_id"].isin(faixas_unicas)]
    log(f"Apos remover faixas ambiguas ............. {len(df):>7,}")

    df["explicit"] = df["explicit"].astype(int)

    df = df[(df["tempo"] > 0) & (df["duration_ms"] > 0)]
    log(f"Apos remover registros invalidos ......... {len(df):>7,}")
    log(f"Total descartado ......................... {n0 - len(df):>7,} "
        f"({100 * (n0 - len(df)) / n0:.1f}%)")

    return df.reset_index(drop=True)


def obter_dados_modelagem(verbose: bool = True):
    """Retorna ``(X, y, df)`` prontos para o treinamento."""
    df = preparar_dados(carregar_base_completa(), verbose=verbose)
    return df[FEATURES].copy(), df[ALVO].copy(), df


def rotulo_amigavel(genero: str) -> str:
    """Converte o rotulo tecnico em um nome legivel, com emoji."""
    return f"{EMOJIS.get(genero, '')} {ROTULOS_PT.get(genero, genero)}".strip()


def garantir_diretorios() -> None:
    """Cria os diretorios de saida do projeto, se necessario."""
    for d in (DIR_MODELO, DIR_FIGURAS, DIR_RESULTADOS):
        d.mkdir(parents=True, exist_ok=True)
