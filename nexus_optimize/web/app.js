/**
 * NEXUS OPTIMIZE · Dashboard E14: porcentajes coherentes, E13/E12 como evidencia histórica
 * Sólo muestra métricas de un informe HTTP medido, nunca simula cifras.
 * Demo M1/E9: dos llamadas con mismos tweets y sin caché; no es un benchmark.
 */
const byId = (id) => document.getElementById(id);
const fmt = new Intl.NumberFormat("es-ES", {maximumFractionDigits: 2});
const fmt0 = new Intl.NumberFormat("es-ES", {maximumFractionDigits: 0});
const fmtPercent = new Intl.NumberFormat("es-ES", {style: "percent", minimumFractionDigits: 2, maximumFractionDigits: 2});
const examples = [
  "I love this film.",
  "The delivery was a disappointment.",
  "Today is a fantastic day.",
  "The new update is not helpful.",
];

let reportE12 = null;
let activeExperiment = "E12";
let demoReady = false;
let demoPending = false;

async function requestJSON(url, opts = {}) {
  const response = await fetch(url, {cache: "no-store", ...opts});
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}
function validPositive(value) { return Number.isFinite(value) && value > 0; }
function pct(before, after) {
  return validPositive(before) && Number.isFinite(after)
    ? 100 * (after / before - 1) : null;
}
function signed(value) {
  return value === null ? "No medido" :
    `${value > 0 ? "+" : ""}${fmt.format(value)} %`;
}
function assessment(value, type) {
  if (value === null) return {text: "No hay datos", outcome: "neutral"};
  const good = type === "latency" ? value < 0 : value > 0;
  const tiny = Math.abs(value) < 0.001;
  return {
    text: `${signed(value)} · ${tiny ? "Sin cambio medible" : good ? "Mejoró" : "Empeoró"}`,
    outcome: tiny ? "neutral" : good ? "good" : "bad",
  };
}
function setBar(idM1, idE9, before, after) {
  const maximum = Math.max(before, after);
  byId(idM1).style.width = `${Math.max(2, before / maximum * 100)}%`;
  byId(idE9).style.width = `${Math.max(2, after / maximum * 100)}%`;
}
function setChange(id, value, metric) {
  const obj = assessment(value, metric);
  byId(id).textContent = obj.text;
  byId(id).dataset.outcome = obj.outcome;
}
function cell(tr, value) {
  const td = document.createElement("td");
  td.textContent = String(value);
  tr.append(td);
  return td;
}
function getRow(n, concurrence) {
  if (!reportE12 || !Array.isArray(reportE12.results)) return null;
  return reportE12.results.find(row =>
    row.tweets === n && row.concurrency === concurrence &&
    (!["E13", "E14"].includes(activeExperiment) || row.cohort === byId("cohort-select").value)) || null;
}
function resetEvidence(message) {
  for (const id of ["results-content", "interpretation-panel", "comparison-table-panel"]) {
    byId(id).hidden = true;
  }
  byId("results-status").textContent = message;
  byId("hero-report-state").textContent = "Sin medición disponible";
  byId("hero-latency").textContent = "—";
  byId("hero-throughput").textContent = "—";
  byId("hero-latency-detail").textContent = "Menos tiempo es mejor";
  byId("hero-throughput-detail").textContent = "Más rendimiento es mejor";
  byId("hero-context").textContent = "No se muestran cifras inventadas.";
  byId("paired-metric-detail").hidden = true;
}
function renderComparisonRows(n, chosenConcurrency) {
  const body = byId("comparison-rows");
  body.replaceChildren();
  if (!reportE12) return;
  const rows = reportE12.results.filter(value => value.tweets === n &&
    (!["E13", "E14"].includes(activeExperiment) || value.cohort === byId("cohort-select").value))
    .sort((a, b) => a.concurrency - b.concurrency);
  for (const item of rows) {
    const tr = document.createElement("tr");
    if (item.concurrency === chosenConcurrency) tr.classList.add("selected-row");
    const eff = item.effective_concurrency ?? "—";
    cell(tr, `${item.concurrency} (${eff} ef.)`);
    if (item.phase === "excluded" ||
        (activeExperiment === "E12" && item.effective_concurrency < item.concurrency)) {
      cell(tr, "No aplica");
      cell(tr, "No aplica");
      cell(tr, "Solicitudes insuficientes");
      cell(tr, "—");
      cell(tr, "—");
      cell(tr, "No se midió");
    } else {
      cell(tr, `${fmt.format(item.m1_p95_ms)} ms`);
      cell(tr, `${fmt.format(item.e9_p95_ms)} ms`);
      const latencyResult = assessment(
        activeExperiment === "E14" ? item.p95_change_pct :
          pct(item.m1_p95_ms, item.e9_p95_ms), "latency");
      const lc = cell(tr, latencyResult.text);
      lc.dataset.outcome = latencyResult.outcome;
      cell(tr, fmt0.format(item.m1_throughput_tweets_s));
      cell(tr, fmt0.format(item.e9_throughput_tweets_s));
      const rateResult = assessment(
        activeExperiment === "E14" ? item.throughput_gain_pct :
          pct(item.m1_throughput_tweets_s, item.e9_throughput_tweets_s),
        "throughput");
      const tc = cell(tr, rateResult.text);
      tc.dataset.outcome = rateResult.outcome;
    }
    body.append(tr);
  }
  byId("comparison-table-panel").hidden = rows.length === 0;
}
function renderEvidence() {
  const n = Number(byId("size-select").value);
  const concurrency = Number(byId("concurrency-select").value);
  const row = getRow(n, concurrency);
  const invalidConcurrency = Boolean(row && activeExperiment === "E12" &&
    row.effective_concurrency < row.concurrency);
  if (!row || row.phase === "excluded" || invalidConcurrency) {
    const explanation = invalidConcurrency
      ? "No aplica: el volumen no generó suficientes solicitudes para evaluar la concurrencia seleccionada."
      : row?.reason || "No hay resultados para esta combinación en el informe.";
    resetEvidence(explanation);
    byId("hero-report-state").textContent = "No aplica: no se midió";
    renderComparisonRows(n, concurrency);
    return;
  }
  const latency = activeExperiment === "E14" ? row.p95_change_pct :
    pct(row.m1_p95_ms, row.e9_p95_ms);
  const throughput = activeExperiment === "E14" ? row.throughput_gain_pct :
    pct(row.m1_throughput_tweets_s, row.e9_throughput_tweets_s);
  if (latency === null || throughput === null) {
    resetEvidence("El reporte contiene métricas no comparables.");
    return;
  }
  byId("results-status").textContent =
    `${fmt0.format(n)} tweets reales · ${row.effective_concurrency} solicitudes simultáneas medidas` +
    (row.phase === "exploratory" ? " · Prueba exploratoria (no forma parte de la validación principal)." : ".");
  byId("latency-before").textContent = `${fmt.format(row.m1_p95_ms)} ms`;
  byId("latency-after").textContent = `${fmt.format(row.e9_p95_ms)} ms`;
  byId("throughput-before").textContent =
    `${fmt0.format(row.m1_throughput_tweets_s)} tweets/s`;
  byId("throughput-after").textContent =
    `${fmt0.format(row.e9_throughput_tweets_s)} tweets/s`;
  setBar("latency-original-bar", "latency-optimized-bar", row.m1_p95_ms, row.e9_p95_ms);
  setBar("throughput-original-bar", "throughput-optimized-bar",
    row.m1_throughput_tweets_s, row.e9_throughput_tweets_s);
  setChange("latency-delta", latency, "latency");
  setChange("throughput-delta", throughput, "throughput");
  // Las dos metodologías estadísticas permanecen en los reportes técnicos,
  // pero la pantalla pública usa exclusivamente el cambio entre las medianas visibles.
  const pairedNote = byId("paired-metric-detail");
  pairedNote.hidden = true;
  pairedNote.textContent = "";
  const outcome = row.phase === "exploratory"
    ? "Esta medición sirve para explorar el comportamiento, pero no determina la conclusión general."
    : row.result === "improved"
      ? "La ruta optimizada mejoró la velocidad y cumplió los criterios definidos para esta carga."
      : row.result === "regressed"
        ? "En esta carga la optimización empeoró; también mostramos los resultados desfavorables."
        : row.result === "inconclusive"
          ? "La medición no fue suficientemente estable para concluir que existe una mejora."
          : "No se demostró una mejora suficiente en este escenario.";
  byId("scenario-interpretation").textContent =
    `${outcome} Las dos rutas utilizan el mismo clasificador y no emplean caché.`;
  byId("report-method").textContent =
    `${reportE12.methodology.repeats} rondas de medición HTTP en el mismo servidor local; ` +
    (["E13", "E14"].includes(activeExperiment)
      ? "dos muestras independientes de tweets y ejecución alternada. "
      : "medición previa con limitaciones de concurrencia. ") +
    "La tasa de tweets/s se compara mediante las medianas mostradas. " +
    "El Algoritmo Genético se evaluó, pero no explica la mejora del cálculo fusionado.";
  const exact = row.labels_identical;
  byId("validation-title").textContent =
    exact ? "Predicciones idénticas" : "Predicciones no equivalentes";
  byId("validation-title").dataset.outcome = exact ? "good" : "bad";
  byId("validation-detail").textContent =
    reportE12.decision === "candidate_validated"
      ? "Mejora validada en las cargas principales, con controles de estabilidad."
      : "La mejora general todavía no está validada en todas las cargas principales.";

  byId("hero-latency").textContent = signed(latency);
  byId("hero-throughput").textContent = signed(throughput);
  byId("hero-latency").dataset.outcome = assessment(latency, "latency").outcome;
  byId("hero-throughput").dataset.outcome = assessment(throughput, "throughput").outcome;
  byId("hero-latency-detail").textContent =
    `${fmt.format(row.m1_p95_ms)} → ${fmt.format(row.e9_p95_ms)} ms`;
  byId("hero-throughput-detail").textContent =
    `${fmt0.format(row.m1_throughput_tweets_s)} → ` +
    `${fmt0.format(row.e9_throughput_tweets_s)} tweets/s`;
  byId("hero-context").textContent =
    `${fmt0.format(n)} tweets únicos · concurrencia efectiva ${row.effective_concurrency}`;
  byId("hero-report-state").textContent = reportE12.decision === "candidate_validated"
    ? "Mejora validada" : "Mejora global no validada";

  renderComparisonRows(n, concurrency);
  for (const id of ["results-content", "interpretation-panel", "comparison-table-panel"]) {
    byId(id).hidden = false;
  }
}
async function loadEvidence() {
  byId("reload-results").disabled = true;
  byId("results-status").textContent = "Cargando resultados medidos…";
  try {
    const reconciled = await requestJSON("/api/metrics-closure/results");
    let report = null;
    if (reconciled.schema_version === 14 && reconciled.status === "measured") {
      report = reconciled;
      activeExperiment = "E14";
    } else {
      // E14 aún no generado: E13 y E12 se presentan únicamente como históricos.
      const latest = await requestJSON("/api/global-http/results");
      if (latest.schema_version === 13 && latest.status === "measured") {
        report = latest;
        activeExperiment = "E13";
      } else {
        const previous = await requestJSON("/api/real-http/results");
        if (previous.schema_version === 12 && previous.status === "measured") {
          report = previous;
          activeExperiment = "E12";
        }
      }
    }
    if (!report || !Array.isArray(report.results) || !report.results.length) {
      reportE12 = null;
      resetEvidence("Aún no hay mediciones disponibles para comparar las dos versiones.");
      byId("global-status").textContent = "No hay resultados medidos disponibles.";
      return;
    }
    reportE12 = report;
    byId("cohort-field").hidden = !["E13", "E14"].includes(activeExperiment);
    if (activeExperiment === "E14" || activeExperiment === "E13") {
      const summary = report.summary;
      const conclusion = report.decision === "candidate_validated"
        ? "Mejora validada en las cargas principales."
        : "No se validó una mejora general en todas las cargas principales.";
      byId("global-status").textContent =
        `${summary.primary_improved} de ${summary.primary_total} comparaciones principales mejoraron; ` +
        `${summary.primary_regressed} registraron regresiones. ` +
        `${summary.excluded} combinaciones no pudieron medirse con la concurrencia solicitada. ` +
        `${conclusion} Se utilizaron dos muestras independientes.` +
        (activeExperiment === "E14"
          ? (report.decision_changed
              ? " La revisión de las métricas cambió la conclusión anterior."
              : " Los porcentajes están conciliados con los valores mostrados.")
          : " Los porcentajes aún están pendientes de la revisión final.");
    } else {
      byId("global-status").textContent =
        "Resultados históricos disponibles, pero con limitaciones de concurrencia. " +
        "Las mediciones más recientes todavía no están disponibles.";
    }
    renderEvidence();
  } catch {
    reportE12 = null;
    resetEvidence("No se pudieron verificar los resultados experimentales.");
    byId("global-status").textContent =
      "No se muestran datos si el reporte es inválido o no coincide con la medición de origen.";
  } finally {
    byId("reload-results").disabled = false;
  }
}
async function loadStatus() {
  try {
    const [status, historical] = await Promise.all([
      requestJSON("/api/status"), requestJSON("/api/history"),
    ]);
    demoReady = Boolean(status.model_ready && status.e9_api_enabled);
    byId("demo-model-status").textContent = demoReady
      ? (status.public_demo
        ? "Comparación M1 y M2 disponible en el demostrador público (máximo 8 tweets)"
        : "Comparación M1 y M2 disponible en el servidor local")
      : "La comparación no está habilitada en este servidor";
    byId("f1-macro").textContent =
      Number.isFinite(historical.f1_macro) && historical.f1_macro >= 0 && historical.f1_macro <= 1
        ? fmtPercent.format(historical.f1_macro) : "No disponible";
  } catch {
    demoReady = false;
    byId("demo-model-status").textContent = "API no disponible";
    byId("f1-macro").textContent = "No disponible";
  }
  byId("demo-run").disabled = !demoReady;
}
async function runDemo(event) {
  event.preventDefault();
  if (!demoReady || demoPending) return;
  const texts = byId("demo-texts").value.split(/\r?\n/)
    .map(value => value.trim()).filter(Boolean);
  if (texts.length < 1 || texts.length > 8 || texts.some(text => text.length > 500)) {
    byId("demo-response").textContent =
      "Introduce entre 1 y 8 tweets no vacíos, máximo 500 caracteres por línea.";
    byId("demo-detail").hidden = true;
    return;
  }
  demoPending = true;
  byId("demo-run").disabled = true;
  byId("demo-detail").hidden = true;
  byId("demo-response").textContent = "Comparando la clasificación original y optimizada…";
  try {
    const invoke = (strategy) => requestJSON("/api/pipeline/batch", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({texts, language: "en", strategy}),
    });
    const m1 = await invoke("m1");
    const e9 = await invoke("e9");
    if (m1.strategy !== "m1" || e9.strategy !== "e9" ||
        !Array.isArray(m1.sentiments) || !Array.isArray(e9.sentiments) ||
        m1.sentiments.length !== texts.length ||
        e9.sentiments.length !== texts.length) {
      throw new Error("Respuesta de predicción incompleta.");
    }
    const identical = m1.sentiments.every(
      (sentiment, idx) => sentiment === e9.sentiments[idx]);
    byId("demo-equivalence").textContent = identical
      ? `${texts.length} / ${texts.length} idénticas` : "Existen diferencias";
    byId("demo-equivalence").dataset.outcome = identical ? "good" : "bad";
    byId("demo-reference-ms").textContent = `${fmt.format(m1.processing_ms)} ms`;
    byId("demo-optimized-ms").textContent = `${fmt.format(e9.processing_ms)} ms`;
    byId("demo-path").textContent = "Sin caché · cálculo fusionado";
    byId("demo-labels").replaceChildren();
    for (const [idx, sentiment] of m1.sentiments.entries()) {
      const item = document.createElement("li");
      item.textContent = `Tweet ${idx + 1}: ${sentiment}`;
      byId("demo-labels").append(item);
    }
    byId("demo-response").textContent = identical
      ? "Las dos rutas generaron las mismas etiquetas en esta prueba."
      : "Atención: las predicciones no coincidieron.";
    byId("demo-detail").hidden = false;
  } catch (error) {
    byId("demo-response").textContent =
      `No se pudo completar la prueba (${error.message}).`;
  } finally {
    demoPending = false;
    byId("demo-run").disabled = !demoReady;
  }
}
byId("size-select").addEventListener("change", renderEvidence);
byId("concurrency-select").addEventListener("change", renderEvidence);
byId("cohort-select").addEventListener("change", renderEvidence);
byId("reload-results").addEventListener("click", loadEvidence);
byId("sample-unique").addEventListener("click", () => {
  byId("demo-texts").value = examples.join("\n");
  byId("demo-response").textContent = "Ejemplos cargados. Ejecuta la comparación.";
  byId("demo-detail").hidden = true;
});
byId("demo-form").addEventListener("submit", runDemo);
await Promise.all([loadEvidence(), loadStatus()]);

/** Seis pestañas informativas de M2; accesibles con teclado sin pedir datos al servidor. */
const infoTabs = Array.from(document.querySelectorAll('#informacion [role="tab"]'));
function activateInfoTab(tab, focus = false) {
  if (!infoTabs.includes(tab)) return;
  for (const button of infoTabs) {
    const active = button === tab;
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
    const panel = document.getElementById(button.getAttribute("aria-controls"));
    if (panel) panel.hidden = !active;
  }
  if (focus) tab.focus();
}
for (const [index, tab] of infoTabs.entries()) {
  tab.addEventListener("click", () => activateInfoTab(tab));
  tab.addEventListener("keydown", (event) => {
    let destination = null;
    if (event.key === "ArrowRight") destination = infoTabs[(index + 1) % infoTabs.length];
    if (event.key === "ArrowLeft") destination = infoTabs[(index - 1 + infoTabs.length) % infoTabs.length];
    if (event.key === "Home") destination = infoTabs[0];
    if (event.key === "End") destination = infoTabs[infoTabs.length - 1];
    if (destination) {
      event.preventDefault();
      activateInfoTab(destination, true);
    }
  });
}
/** Trivia local: preguntas formativas, sin persistencia ni transmisión de respuestas. */
byId("trivia-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const answers = new FormData(event.currentTarget);
  const correct = {q1: "b", q2: "a", q3: "c"};
  let score = 0;
  for (const [question, expected] of Object.entries(correct)) {
    if (answers.get(question) === expected) score += 1;
  }
  byId("trivia-result").textContent =
    `${score} de 3 correctas. ` +
    (score === 3
      ? "Comprendiste cómo se relacionan M1, M2 y el AG."
      : "Repasa: M2 acelera los cálculos sin reentrenar; p95 menor es mejor; el AG no superó el cálculo fusionado.");
});
