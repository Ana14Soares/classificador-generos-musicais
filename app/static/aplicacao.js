/* ==========================================================================
   aplicacao.js — lógica de interface do classificador de gêneros musicais
   Trabalho Final de Probabilidade e Estatística — IFCE
   ========================================================================== */

(function () {
  "use strict";

  const { features, info } = window.CONFIG;

  const $ = (id) => document.getElementById(id);

  const formulario       = $("formulario");
  const btnPrever        = $("btn-prever");
  const btnExemplo       = $("btn-exemplo");
  const btnLimpar        = $("btn-limpar");
  const seletorGenero    = $("seletor-genero");
  const avisoErro        = $("aviso-erro");
  const estadoVazio      = $("estado-vazio");
  const resultadoBloco   = $("resultado-conteudo");
  const blocoComparacao  = $("comparacao");
  const faixaCarregada   = $("faixa-carregada");

  /** Gênero real da faixa carregada da base (null quando o usuário digitou). */
  let generoRealAtual = null;

  /* ------------------------------------------------------------ utilidades */

  /** Formata milissegundos como "3min 30s". */
  function formatarDuracao(ms) {
    const total = Math.round(ms / 1000);
    const min = Math.floor(total / 60);
    const seg = total % 60;
    return `≈ ${min}min ${String(seg).padStart(2, "0")}s`;
  }

  /**
   * Formata um percentual evitando exibir "100,00%" ou "0,00%" por
   * arredondamento: o Naive Bayes produz probabilidades extremas, mas nunca
   * exatamente 1 ou 0, e mostrar certeza absoluta seria enganoso.
   */
  function formatarPercentual(p) {
    if (p >= 99.995) return "> 99,99%";
    if (p > 0 && p < 0.005) return "< 0,01%";
    return `${p.toFixed(2).replace(".", ",")}%`;
  }

  /** Limita um valor ao intervalo válido da feature. */
  function limitar(valor, min, max) {
    return Math.min(max, Math.max(min, valor));
  }

  /** Arredonda conforme o passo da feature, evitando ruído de ponto flutuante. */
  function arredondar(valor, passo) {
    const casas = (String(passo).split(".")[1] || "").length;
    return Number(valor.toFixed(casas));
  }

  function atualizarAuxiliarDuracao() {
    const aux = $("auxiliar-duration");
    if (aux) aux.textContent = formatarDuracao(Number($("campo-duration_ms").value));
  }

  /* ----------------------------------------- sincronização slider <-> input */

  features.forEach((f) => {
    if (f === "explicit") return;

    const deslizante = $(`faixa-${f}`);
    const numero = $(`campo-${f}`);
    if (!deslizante || !numero) return;

    // Rolar a página com o cursor sobre um controle deslizante ou campo
    // numérico altera o valor silenciosamente e corrompe a previsão. Aqui o
    // valor anterior e restaurado logo apos a rolagem, sem bloquear o scroll
    // da pagina (o que prejudicaria a navegacao em um formulario longo).
    const protegerDaRoda = (elemento) => {
      elemento.addEventListener("wheel", () => {
        const anterior = elemento.value;
        if (document.activeElement === elemento) elemento.blur();
        requestAnimationFrame(() => {
          if (elemento.value !== anterior) {
            elemento.value = anterior;
            deslizante.value = anterior;
            numero.value = anterior;
            if (f === "duration_ms") atualizarAuxiliarDuracao();
          }
        });
      }, { passive: true });
    };
    protegerDaRoda(deslizante);
    protegerDaRoda(numero);

    deslizante.addEventListener("input", () => {
      numero.value = deslizante.value;
      numero.classList.remove("invalido");
      if (f === "duration_ms") atualizarAuxiliarDuracao();
    });

    numero.addEventListener("input", () => {
      const bruto = parseFloat(numero.value);
      if (Number.isNaN(bruto)) {
        numero.classList.add("invalido");
        return;
      }
      numero.classList.remove("invalido");
      deslizante.value = limitar(bruto, info[f].min, info[f].max);
      if (f === "duration_ms") atualizarAuxiliarDuracao();
    });

    // Ao sair do campo, corrige valores fora do intervalo permitido.
    numero.addEventListener("blur", () => {
      const bruto = parseFloat(numero.value);
      if (Number.isNaN(bruto)) {
        numero.value = info[f].padrao;
      } else {
        numero.value = arredondar(limitar(bruto, info[f].min, info[f].max), info[f].passo);
      }
      numero.classList.remove("invalido");
      deslizante.value = numero.value;
      if (f === "duration_ms") atualizarAuxiliarDuracao();
    });
  });

  // Alternador de conteúdo explícito
  const checkExplicito = $("campo-explicit-check");
  if (checkExplicito) {
    checkExplicito.addEventListener("change", () => {
      $("campo-explicit").value = checkExplicito.checked ? "1" : "0";
      $("texto-explicit").textContent = checkExplicito.checked ? "Sim" : "Não";
    });
  }

  atualizarAuxiliarDuracao();

  /* ------------------------------------------------- preenchimento do form */

  function definirValores(valores) {
    features.forEach((f) => {
      if (!(f in valores)) return;
      const valor = Number(valores[f]);

      if (f === "explicit") {
        const ligado = valor >= 0.5;
        $("campo-explicit").value = ligado ? "1" : "0";
        $("campo-explicit-check").checked = ligado;
        $("texto-explicit").textContent = ligado ? "Sim" : "Não";
        return;
      }

      const ajustado = arredondar(limitar(valor, info[f].min, info[f].max), info[f].passo);
      $(`campo-${f}`).value = ajustado;
      $(`faixa-${f}`).value = ajustado;
      $(`campo-${f}`).classList.remove("invalido");
    });
    atualizarAuxiliarDuracao();
  }

  function restaurarPadroes() {
    const padroes = {};
    features.forEach((f) => { padroes[f] = info[f].padrao; });
    definirValores(padroes);

    generoRealAtual = null;
    faixaCarregada.classList.add("oculto");
    blocoComparacao.classList.add("oculto");
    avisoErro.classList.add("oculto");
    resultadoBloco.classList.add("oculto");
    estadoVazio.classList.remove("oculto");
  }

  btnLimpar.addEventListener("click", restaurarPadroes);

  /* --------------------------------------------- carregar faixa de exemplo */

  btnExemplo.addEventListener("click", async () => {
    const genero = seletorGenero.value;
    const url = genero ? `/exemplo?genero=${encodeURIComponent(genero)}` : "/exemplo";

    btnExemplo.disabled = true;
    try {
      const resposta = await fetch(url);
      const dados = await resposta.json();
      if (!dados.ok) throw new Error(dados.erro || "Não foi possível carregar o exemplo.");

      definirValores(dados.valores);
      generoRealAtual = dados.rotulo_real;

      $("faixa-nome").textContent = dados.nome;
      $("faixa-artista").textContent = `— ${dados.artista}`;
      $("faixa-genero").textContent = dados.rotulo_real;
      faixaCarregada.classList.remove("oculto");
      avisoErro.classList.add("oculto");
    } catch (erro) {
      mostrarErro(erro.message);
    } finally {
      btnExemplo.disabled = false;
    }
  });

  /* -------------------------------------------------------------- previsão */

  function mostrarErro(mensagem) {
    avisoErro.textContent = mensagem;
    avisoErro.classList.remove("oculto");
  }

  function definirCarregando(ativo) {
    btnPrever.disabled = ativo;
    btnPrever.querySelector(".btn-texto-conteudo").textContent =
      ativo ? "Calculando…" : "Prever gênero";
    btnPrever.querySelector(".carregando").classList.toggle("oculto", !ativo);
  }

  function renderizarResultado(dados) {
    estadoVazio.classList.add("oculto");
    resultadoBloco.classList.remove("oculto");

    $("resultado-emoji").textContent = dados.emoji;
    $("resultado-genero").textContent = dados.rotulo;
    $("resultado-confianca").textContent = formatarPercentual(dados.confianca);
    $("texto-confianca").textContent = dados.texto_confianca;

    const barra = $("medidor-barra");
    barra.className = "medidor-barra " + dados.nivel_confianca;
    // Pequeno atraso para a transição CSS ser perceptível.
    requestAnimationFrame(() => { barra.style.width = `${dados.confianca}%`; });

    // Comparação com o gênero real, quando a faixa veio da base.
    if (generoRealAtual) {
      const acertou = generoRealAtual === dados.rotulo;
      blocoComparacao.className = "comparacao " + (acertou ? "acerto" : "erro");
      $("comparacao-texto").textContent = acertou
        ? `✓ O modelo acertou: a faixa realmente é ${generoRealAtual}.`
        : `✗ O gênero real desta faixa é ${generoRealAtual}.`;
      blocoComparacao.classList.remove("oculto");
    } else {
      blocoComparacao.classList.add("oculto");
    }

    // Ranking de probabilidades
    const lista = $("lista-ranking");
    lista.innerHTML = "";
    dados.ranking.forEach((item, indice) => {
      const li = document.createElement("li");
      li.className = "item-ranking" + (indice === 0 ? " destaque" : "");
      li.innerHTML = `
        <div class="item-ranking-topo">
          <span>${item.emoji}</span>
          <span>${item.rotulo}</span>
        </div>
        <span class="item-ranking-valor">${formatarPercentual(item.percentual)}</span>
        <div class="item-trilha"><div class="item-barra"></div></div>
      `;
      lista.appendChild(li);
      requestAnimationFrame(() => {
        li.querySelector(".item-barra").style.width = `${item.percentual}%`;
      });
    });
  }

  formulario.addEventListener("submit", async (evento) => {
    evento.preventDefault();
    avisoErro.classList.add("oculto");

    // Monta o payload a partir dos campos do formulário.
    const payload = {};
    let invalido = null;

    for (const f of features) {
      const campo = $(`campo-${f}`);
      const valor = parseFloat(campo.value);
      if (Number.isNaN(valor)) {
        campo.classList.add("invalido");
        invalido = invalido || `Informe um valor numérico para "${info[f].rotulo}".`;
        continue;
      }
      payload[f] = valor;
    }

    if (invalido) {
      mostrarErro(invalido);
      return;
    }

    definirCarregando(true);
    try {
      const resposta = await fetch("/prever", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const dados = await resposta.json();

      if (!dados.ok) {
        mostrarErro(dados.erro || "Não foi possível realizar a previsão.");
        return;
      }
      renderizarResultado(dados);
    } catch (erro) {
      mostrarErro("Falha de comunicação com o servidor. Verifique se a aplicação está em execução.");
    } finally {
      definirCarregando(false);
    }
  });

  // Alterar manualmente qualquer campo invalida a comparação com o gênero real.
  formulario.addEventListener("input", (evento) => {
    if (evento.target.matches(".numero, .deslizante, .alternador-check")) {
      if (generoRealAtual !== null) {
        generoRealAtual = null;
        faixaCarregada.classList.add("oculto");
        blocoComparacao.classList.add("oculto");
      }
    }
  });
})();
