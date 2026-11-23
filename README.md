# 🎵 Classificador de Gêneros Musicais com Naive Bayes

**Trabalho Final — Probabilidade e Estatística**
Curso de Engenharia da Computação — Instituto Federal do Ceará (IFCE)
Professor: Valberto Feitosa

**Equipe 07:** Ana Luiza Soares · Ana Virna Sousa Carvalho · Maria Fernanda Braga Queiroz

Solução completa de *Machine Learning*, da análise da base de dados até a
disponibilização do modelo treinado em uma aplicação web.

---

## 1. Visão geral do projeto

| Item | Descrição |
|---|---|
| **Problema** | Classificação supervisionada multiclasse |
| **Pergunta** | A partir das características acústicas de uma faixa, qual é o seu gênero musical? |
| **Base de dados** | [Spotify Tracks Dataset (Kaggle)](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) |
| **Observações (base original)** | 114.000 faixas |
| **Observações (após pré-processamento)** | 9.435 faixas |
| **Variável-alvo** | `track_genre` (categórica) |
| **Classes** | 10 gêneros musicais |
| **Variáveis preditoras** | 12 atributos |
| **Algoritmo** | Naive Bayes (`CategoricalNB` com discretização por quantis) |
| **Acurácia no teste** | **82,83 %** |
| **F1-Score macro** | **0,8261** |

> O acaso acertaria 10 % em um problema balanceado de 10 classes.
> O modelo é **8,3× melhor que o acaso**.

---

## 2. Estrutura de pastas

```
2-Trabalho-Final/
├── README.md                        ← este arquivo (documentação do projeto)
├── requirements.txt                 ← dependências com versões fixadas
│
├── dados/
│   └── raw/dataset.csv              ← base original (114.000 faixas)
│
├── src/                             ← código de análise e treinamento
│   ├── preparacao.py                ← configuração central e pré-processamento
│   ├── 00_justificativa_subconjunto.py  ← experimento que justifica as 10 classes
│   ├── 00b_selecao_de_features.py   ← experimento que justifica as 12 features
│   ├── 01_analise_exploratoria.py   ← AED + geração das figuras
│   └── 02_treinar_modelo.py         ← treino, avaliação e salvamento do modelo
│
├── modelo/                          ← artefatos gerados pelo treinamento
│   ├── modelo_naive_bayes.joblib    ← modelo treinado (joblib)
│   ├── metadados_modelo.json        ← métricas, features, hiperparâmetros
│   └── exemplos.json                ← faixas reais usadas na interface
│
├── app/                             ← aplicação web
│   ├── app.py                       ← back-end (Flask)
│   ├── templates/index.html         ← front-end (HTML)
│   └── static/
│       ├── estilo.css               ← estilos (CSS)
│       └── aplicacao.js             ← interatividade (JavaScript)
│
├── relatorio/
│   ├── RELATORIO.md                 ← relatório técnico completo
│   └── figuras/                     ← 10 figuras geradas pelos scripts
│
└── resultados/                      ← tabelas (.csv) geradas pelos scripts
```

---

## 3. Como executar

### 3.1. Preparar o ambiente

```bash
cd 2-Trabalho-Final
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3.2. Obter a base de dados

O arquivo `dados/raw/dataset.csv` já acompanha o projeto. Caso precise baixá-lo
novamente, ele está disponível em
[kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
e deve ser salvo em `dados/raw/dataset.csv`.

### 3.3. Executar o pipeline

```bash
python src/00_justificativa_subconjunto.py   # (opcional) justifica a escolha das classes
python src/00b_selecao_de_features.py        # (opcional) justifica a escolha das features
python src/01_analise_exploratoria.py        # análise exploratória + figuras
python src/02_treinar_modelo.py              # treina, avalia e salva o modelo
```

### 3.4. Iniciar a aplicação web

```bash
python app/app.py
```

Acesse **http://127.0.0.1:5000**

> **macOS:** a porta 5000 costuma estar ocupada pelo serviço *AirPlay Receiver*.
> Nesse caso use `PORT=5050 python app/app.py` e acesse `http://127.0.0.1:5050`.

---

## 4. A base de dados

### 4.1. Origem e características

O *Spotify Tracks Dataset* reúne **114.000 faixas** distribuídas em
**114 gêneros musicais**, com exatamente **1.000 faixas por gênero** — ou seja,
a base original é perfeitamente balanceada.

Cada faixa possui 20 colunas: 4 de identificação (`track_id`, `artists`,
`album_name`, `track_name`), 15 atributos numéricos/booleanos e a variável-alvo
`track_genre`.

### 4.2. Variável-alvo e classes

A variável-alvo é `track_genre`. Dos 114 gêneros originais, o trabalho utiliza
**10 classes acusticamente contrastantes**:

| Gênero | Rótulo na interface | Gênero | Rótulo na interface |
|---|---|---|---|
| `classical` | Música Clássica | `jazz` | Jazz |
| `comedy` | Comédia / Falado | `opera` | Ópera |
| `death-metal` | Death Metal | `reggaeton` | Reggaeton |
| `forro` | Forró | `sleep` | Sleep / Relaxamento |
| `hip-hop` | Hip-Hop | `techno` | Techno |

A justificativa completa dessa escolha está na
[seção 3 do relatório](relatorio/RELATORIO.md) e é reproduzível pelo script
`src/00_justificativa_subconjunto.py`.

---

## 5. Variáveis preditoras

As 12 variáveis usadas no treinamento, todas estatisticamente significativas
(ANOVA, *p* < 0,001):

| # | Variável | Descrição | Razão F |
|---|---|---|---|
| 1 | `speechiness` | Proporção de palavras faladas | 3.391,5 |
| 2 | `danceability` | Adequação para dançar | 2.398,5 |
| 3 | `acousticness` | Confiança de que a faixa é acústica | 2.339,3 |
| 4 | `loudness` | Volume médio em dB | 1.827,6 |
| 5 | `energy` | Intensidade e atividade percebidas | 1.558,7 |
| 6 | `instrumentalness` | Ausência de vocais | 1.120,0 |
| 7 | `valence` | Positividade musical | 965,8 |
| 8 | `liveness` | Indício de gravação ao vivo | 592,0 |
| 9 | `explicit` | Letra explícita (0/1) | 539,2 |
| 10 | `popularity` | Índice de popularidade (0–100) | 260,8 |
| 11 | `tempo` | Andamento em BPM | 190,5 |
| 12 | `duration_ms` | Duração em milissegundos | 85,2 |

---

## 6. Resultados

### 6.1. Comparação entre variantes do Naive Bayes

Validação cruzada (5 *folds*) no conjunto de treinamento:

| Modelo | Acurácia | F1-macro |
|---|---|---|
| Linha de base (classe majoritária) | 0,1056 | — |
| `GaussianNB` | 0,5440 | 0,5270 |
| Yeo-Johnson + `GaussianNB` | 0,6435 | 0,6061 |
| **Discretização + `CategoricalNB`** | **0,8035** | **0,8006** |

### 6.2. Desempenho final (conjunto de teste, 1.887 faixas)

| Métrica | Valor |
|---|---|
| Acurácia | **0,8283** |
| Precisão (macro) | 0,8278 |
| Revocação (macro) | 0,8263 |
| F1-Score (macro) | 0,8261 |

Métricas por gênero, matriz de confusão e análise dos erros estão no
[relatório técnico](relatorio/RELATORIO.md).

---

## 7. A aplicação web

### Front-end (`app/templates/index.html`, `app/static/`)

* Página HTML com formulário contendo os **12 campos** correspondentes às
  variáveis do treinamento;
* Cada campo tem **controle deslizante + campo numérico sincronizados**, com
  faixa de valores válidos e descrição em *tooltip*;
* Botão **“Prever gênero”**;
* Botão **“Carregar faixa real da base”**, que preenche o formulário com uma
  música verdadeira e permite comparar a previsão com o gênero real;
* Exibição do resultado com o gênero previsto, a confiança do modelo e o
  **ranking de probabilidades** de todas as 10 classes;
* Painel “Sobre o modelo” com as métricas de avaliação;
* Layout responsivo (desktop e celular) e validação de entrada.

### Back-end (`app/app.py`)

| Rota | Método | Função |
|---|---|---|
| `/` | GET | Renderiza a página com o formulário |
| `/prever` | POST | Valida a entrada, aplica o modelo e devolve a previsão |
| `/exemplo` | GET | Devolve uma faixa real da base (opcionalmente filtrada por gênero) |
| `/sobre` | GET | Devolve os metadados do modelo em JSON |

O back-end carrega o modelo salvo **uma única vez**, na inicialização, valida
tipo e intervalo de todos os campos e aceita tanto JSON quanto dados de
formulário.

---

## 8. Decisões técnicas documentadas

1. **Por que 10 gêneros e não 114?**
   Com 114 classes o mesmo modelo atinge 27,3 % de acurácia. O experimento em
   `src/00_justificativa_subconjunto.py` mostra que o gargalo é a sobreposição
   entre rótulos comerciais, não o algoritmo.

2. **Por que `CategoricalNB` em vez de `GaussianNB`?**
   A análise exploratória mostrou forte assimetria nas variáveis
   (`duration_ms`: +8,12; `speechiness`: +2,64), violando a hipótese de
   normalidade. A discretização por quantis elimina essa suposição e eleva o
   F1-macro de 0,527 para 0,801.

3. **Por que 12 variáveis e não 15?**
   O experimento em `src/00b_selecao_de_features.py` mostra que `explicit` e
   `popularity` elevam o F1-macro de 0,730 para 0,814, enquanto `key`, `mode` e
   `time_signature` acrescentam apenas +0,0006 — ruído. Foram excluídas por
   parcimônia.

4. **Remoção de faixas ambíguas.**
   24.259 `track_id` aparecem em mais de um gênero na base original. Dentro do
   recorte adotado, essas faixas foram removidas: são rótulos contraditórios
   para o mesmo ponto do espaço de atributos.

5. **Sobre as probabilidades exibidas.**
   O Naive Bayes assume independência condicional entre os atributos, mas
   `energy` e `loudness` têm correlação de Spearman de 0,72. A consequência é
   conhecida: as probabilidades ficam super-confiantes. A interface declara
   isso explicitamente e as trata como ordem de preferência, não como
   probabilidades calibradas.

---

## 9. Tecnologias utilizadas

Python 3 · scikit-learn · Pandas · NumPy · Joblib · Matplotlib · Seaborn ·
Flask · HTML · CSS · JavaScript
