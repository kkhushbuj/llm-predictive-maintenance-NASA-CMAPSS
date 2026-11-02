const ICONS = {
  back: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M19 12H5M11 18l-6-6 6-6"/></svg>',
  spark: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/></svg>',
  send: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6"/></svg>',
};

// A repeating ECG-style trace; the pulse speeds up and changes color with risk.
function ecgSVG(tier) {
  let d = "M0 40";
  for (let i = 0; i < 4; i++) {
    const x = i * 150;
    d += ` H${x + 40} l8 0 l8 -22 l10 44 l8 -22 l8 0 H${x + 150}`;
  }
  const speed = tier === "high" ? 1.8 : tier === "medium" ? 2.8 : 4;
  return `<svg class="ecg" viewBox="0 0 600 80" preserveAspectRatio="none" aria-hidden="true" style="--ecg:var(--risk-${tier});--ecg-speed:${speed}s">
    <path class="base" d="${d}"/><path pathLength="1" d="${d}"/></svg>`;
}

const EngineView = (() => {
  let engine = null;
  let subset = "FD001";
  let sensorRows = null;
  let activeSensor = null;
  let chatMessages = [];
  let chatLoading = false;
  let analysisRevealed = false;

  const STAGES = [
    "Comparing SHAP attribution against the literature knowledge base...",
    "Running independent critique review...",
    "Finalizing verdict...",
  ];

  async function render(unit, engineSubset) {
    subset = engineSubset || "FD001";
    const app = document.getElementById("app");
    app.innerHTML = `
      <a href="#/${subset === "FD001" ? "" : subset}" class="back-link">${ICONS.back} Back to fleet overview</a>
      <div id="engine-content" aria-busy="true">
        <div class="skeleton" style="height:92px;width:100%;margin-bottom:16px;"></div>
        <div class="skeleton" style="height:320px;width:100%;"></div>
      </div>
    `;

    let engineData, sensors;
    try {
      [engineData, sensors] = await Promise.all([
        API.engine(unit, subset),
        API.engineSensors(unit, subset).catch(() => null),
      ]);
    } catch (err) {
      document.getElementById("engine-content").innerHTML = `
        <div class="state-block error" role="alert">
          <h2>Engine ${unit} not found</h2>
          <p>There is no engine ${unit} in the ${subset} test set, or the API couldn't be reached.</p>
          <a class="btn secondary" href="#/${subset === "FD001" ? "" : subset}">Back to fleet overview</a>
        </div>`;
      return;
    }

    engine = engineData;
    sensorRows = sensors;
    activeSensor = engine.shap?.top_sensors?.[0]?.sensor ?? null;
    // Only the pre-flagged top-15 (engine.deep_analysis) auto-display their
    // analysis; every other engine requires an explicit click, even if it
    // happens to already have a precomputed result on disk from a prior
    // on-demand run.
    analysisRevealed = Boolean(engine.deep_analysis);
    try {
      renderContent();
    } catch (err) {
      console.error("Failed to render engine detail:", err);
      document.getElementById("engine-content").innerHTML =
        `<div class="panel">Engine ${unit} data loaded, but the page failed to render: ${err.message}. Check the browser console for details.</div>`;
    }
  }

  function riskBadge(score) {
    const tier = Charts.riskTier(score);
    const label = tier === "high" ? "High risk" : tier === "medium" ? "Medium risk" : "Low risk";
    return `<span class="risk-badge ${tier}"><span class="rb-dot"></span>${label}</span>`;
  }

  function renderContent() {
    const topSensors = engine.shap?.top_sensors?.map((s) => s.sensor) ?? [];

    const error = engine.predicted_RUL - engine.true_RUL;
    const errorText = `${error >= 0 ? "+" : "-"}${Math.abs(error).toFixed(1)}`;
    const errorColor =
      Math.abs(error) <= 10 ? "var(--risk-low)" : Math.abs(error) <= 20 ? "var(--risk-medium)" : "var(--risk-high)";
    const trendCaption =
      subset === "FD001"
        ? "Normalized sensor values (min-max scaled to 0–1 from the training data), not raw units."
        : "Normalized sensor values (min-max scaled to 0–1 across the dataset, for display), not raw units. The model itself sees values scaled per operating condition.";

    document.getElementById("engine-content").removeAttribute("aria-busy");
    const shapSummary = engine.shap?.top_sensors?.length
      ? `Top sensors by influence: ${engine.shap.top_sensors.slice(0, 3).map((s) => s.sensor.replace("sensor_", "Sensor ")).join(", ")}.`
      : "";
    document.getElementById("engine-content").innerHTML = `
      <div class="engine-header">
        ${ecgSVG(Charts.riskTier(engine.risk_score))}
        <div>
          <div class="engine-title">
            <h1>Engine ${engine.unit}</h1>
            ${riskBadge(engine.risk_score)}
            ${engine.deep_analysis ? `<span class="badge-deep">${ICONS.spark} Deep-analyzed</span>` : ""}
          </div>
          <p class="engine-sub">Risk rank #${engine.risk_rank} · ${subset}</p>
        </div>
        <div class="metrics-cluster">
          <div class="metric-block">
            <p class="metric-label">Predicted RUL</p>
            <p class="metric-value">${engine.predicted_RUL.toFixed(1)} <small>cycles</small></p>
          </div>
          <div class="metric-block">
            <p class="metric-label">True RUL</p>
            <p class="metric-value" style="color:var(--text-dim)">${engine.true_RUL.toFixed(0)} <small>cycles</small></p>
          </div>
          <div class="metric-block" id="prediction-error">
            <p class="metric-label">Prediction error</p>
            <p class="metric-value" style="color:${errorColor}">${errorText} <small>cycles</small></p>
            <p class="metric-note">Predicted minus true</p>
          </div>
          <div class="gauge-wrap" role="img" aria-label="Risk score ${engine.risk_score.toFixed(0)} out of 100">
            <canvas id="risk-gauge"></canvas>
            <div class="gauge-center">
              <span class="gauge-value">${engine.risk_score.toFixed(0)}</span>
              <span class="gauge-caption">Risk</span>
            </div>
          </div>
        </div>
      </div>

      <div class="detail-grid">
        <div class="detail-col">
          <div class="panel">
            <div class="panel-header">
              <div>
                <h2 class="panel-title">Which sensors drove this prediction</h2>
                <p class="panel-sub">SHAP attribution. Red bars (left) shorten predicted remaining life and raise risk. Green bars (right) lengthen it and lower risk.</p>
              </div>
            </div>
            ${engine.shap ? `<div class="chart-box"><canvas id="shap-chart" role="img" aria-label="Bar chart of sensor contributions to the predicted remaining life. ${shapSummary}"></canvas></div>` : `<p class="text-dim">No SHAP data for this engine.</p>`}
          </div>

          <div class="panel">
            <div class="panel-header">
              <h2 class="panel-title">Sensor trend over engine life</h2>
              ${
                topSensors.length
                  ? `<label class="sr-only" for="sensor-select">Sensor</label><select class="select-input" id="sensor-select">
                      ${topSensors.map((s) => `<option value="${s}" ${s === activeSensor ? "selected" : ""}>${s.replace("sensor_", "Sensor ")}</option>`).join("")}
                    </select>`
                  : ""
              }
            </div>
            ${
              sensorRows && activeSensor
                ? `<div class="chart-box tall"><canvas id="sensor-chart" role="img" aria-label="Line chart of the selected sensor's normalized value over the engine's cycles"></canvas></div>
                   <p class="chart-caption">${trendCaption}</p>`
                : `<div class="skeleton" style="height:220px;"></div>`
            }
          </div>
        </div>

        <div class="detail-col">
          <div id="analysis-slot"></div>
          <div id="chat-slot"></div>
        </div>
      </div>
    `;

    // FD002/FD004 models predict RUL / 125 (RUL_CAP), so their SHAP values are
    // fractions of that cap; FD001's are already in cycles.
    if (engine.shap) Charts.shapChart("shap-chart", engine.shap.top_sensors, subset === "FD001" ? 1 : 125);
    if (sensorRows && activeSensor) Charts.sensorTrendChart("sensor-chart", sensorRows, activeSensor);
    Charts.gaugeChart("risk-gauge", engine.risk_score);

    const sensorSelect = document.getElementById("sensor-select");
    if (sensorSelect) {
      sensorSelect.addEventListener("change", (e) => {
        activeSensor = e.target.value;
        Charts.sensorTrendChart("sensor-chart", sensorRows, activeSensor);
      });
    }

    renderAnalysisSlot();
    renderChatSlot();

    Fx.rise(".engine-header", { y: 14 });
    Fx.rise(".detail-col:first-child > .panel", { delay: 0.12, gap: 0.12 });
    Fx.rise("#chat-slot", { delay: 0.35 });
    const gaugeValue = document.querySelector(".gauge-value");
    if (gaugeValue) Fx.count(gaugeValue, engine.risk_score, 0, { delay: 0.2, duration: 1 });
  }

  function renderAnalysisSlot() {
    const slot = document.getElementById("analysis-slot");
    if (analysisRevealed && engine.agent_analysis) {
      slot.innerHTML = agentAnalysisHTML(engine.agent_analysis);
    } else {
      // Deliberately the same copy/button regardless of whether this engine's
      // analysis is already precomputed behind the scenes — the click
      // experience (see runAnalysis) is now identical either way, so the
      // panel shouldn't hint at a difference the user can't act on.
      slot.innerHTML = `
        <div class="run-analysis-card">
          <div>
            <h2>No deep analysis yet</h2>
            <p>${
              engine.deep_analysis
                ? "This engine is flagged as high-risk but hasn't been analyzed yet."
                : "This engine wasn't in the top 15 highest-risk engines, so it only has the ML risk score. You can still run the full literature cross-check on demand."
            }</p>
          </div>
          <button class="btn" id="run-analysis-btn">${ICONS.spark} Run deep analysis</button>
        </div>
      `;
      document.getElementById("run-analysis-btn").addEventListener("click", runAnalysis);
    }
    Fx.rise("#analysis-slot > *", { delay: 0.2, gap: 0.14 });
  }

  async function runAnalysis() {
    const slot = document.getElementById("analysis-slot");

    // Already precomputed (a prior on-demand run, or full batch coverage on
    // FD002/FD004) — no API call needed, but still show the same loading
    // transition (for a fixed minimum time) so the click always feels the
    // same, instead of sometimes snapping straight to results.
    const alreadyComputed = Boolean(engine.agent_analysis);

    let stage = 0;
    slot.innerHTML = `
      <div class="loading-analysis" role="status" aria-busy="true">
        <div class="spinner" style="width:20px;height:20px;margin-top:2px;"></div>
        <div>
          <p style="font-weight:600;">Running synthesis, hypothesis and critique</p>
          <p class="stage" id="analysis-stage">${STAGES[0]}</p>
          ${alreadyComputed ? "" : `<p class="stage">Usually takes 15–30 seconds.</p>`}
        </div>
      </div>
    `;
    const interval = alreadyComputed
      ? null
      : setInterval(() => {
          stage = Math.min(stage + 1, STAGES.length - 1);
          const el = document.getElementById("analysis-stage");
          if (el) el.textContent = STAGES[stage];
        }, 4000);

    if (alreadyComputed) {
      await new Promise((r) => setTimeout(r, 900));
      analysisRevealed = true;
      renderAnalysisSlot();
      return;
    }

    try {
      const analysis = await API.analyzeEngine(engine.unit, subset);
      engine.agent_analysis = analysis;
      analysisRevealed = true;
      renderAnalysisSlot();
      toast(`Deep analysis complete — verdict: ${analysis.synthesis_result?.verdict}`, "success");
    } catch (err) {
      toast(err.message || "Deep analysis failed.", "error");
      renderAnalysisSlot();
    } finally {
      clearInterval(interval);
    }
  }

  // Saved agent text says "sensor_11"; show "Sensor 11". Display-only — the
  // saved JSON files are untouched.
  const prettySensors = (t) => String(t ?? "").replace(/\bsensor_(\d+)\b/gi, "Sensor $1");
  const safe = (t) => escapeHTML(prettySensors(t));

  function agentAnalysisHTML(a) {
    const s = a.synthesis_result;
    const h = a.hypothesis_result;
    const c = a.critique_result;
    let html = "";

    if (s) {
      html += `
        <div class="agent-card">
          <div class="agent-card-head">
            <span class="agent-card-label">Synthesis agent<span>Compares SHAP against the literature</span></span>
            <span class="verdict-badge ${s.verdict}">${
              s.verdict === "Confirmed" ? "✓ Confirmed by literature" : s.verdict === "Contradicted" ? "✕ Contradicts literature" : "? Novel — no literature coverage"
            }</span>
          </div>
          <p>${safe(s.reasoning)}</p>
          ${(s.matched_sources || []).map((src) => `<span class="source-chip">${safe(src)}</span>`).join("")}
        </div>`;
    }
    if (h) {
      html += `
        <div class="agent-card">
          <div class="agent-card-head"><span class="agent-card-label">Hypothesis agent<span>Proposes a physical explanation</span></span></div>
          <p>${safe(h.hypothesis)}</p>
          <p style="color:var(--text-dim);">${safe(h.physical_reasoning)}</p>
          ${
            h.next_checks?.length
              ? `<p class="eyebrow-sm">Next checks</p>
                 <ul>${h.next_checks.map((n) => `<li>${safe(n)}</li>`).join("")}</ul>`
              : ""
          }
        </div>`;
    }
    if (c) {
      html += `
        <div class="agent-card review">
          <div class="agent-card-head">
            <span class="agent-card-label">Critique agent<span>Independent review</span></span>
            <span class="agree-badge ${c.agrees_with_pipeline ? "agree" : "disagree"}">${c.agrees_with_pipeline ? "Agrees" : "Disagrees"}</span>
          </div>
          <p>${safe(c.critique)}</p>
          ${c.alternative_explanation ? `<p style="color:var(--text-dim);margin-top:8px;"><strong style="color:var(--text)">Alternative explanation: </strong>${safe(c.alternative_explanation)}</p>` : ""}
        </div>`;
    }
    return html;
  }

  function renderChatSlot() {
    const slot = document.getElementById("chat-slot");
    slot.innerHTML = `
      <div class="chat-panel">
        <div class="chat-head">
          <div class="chat-head-icon" aria-hidden="true">TS</div>
          <div class="chat-head-text">
            <h2 class="chat-head-title" style="margin:0">Engine ${engine.unit} assistant</h2>
            <div class="chat-head-sub">Answers grounded in this engine's data</div>
          </div>
        </div>
        <div class="chat-body" id="chat-body">
          ${
            chatMessages.length === 0
              ? `<div class="chat-welcome">
                   <div class="chat-welcome-title">Hi! Ask me about Engine ${engine.unit}.</div>
                   <div class="chat-welcome-sub">Try one of these:</div>
                   <div class="chat-chips">
                     ${CHAT_SUGGESTIONS.map((s) => `<button class="chat-chip" data-q="${escapeHTML(s)}">${escapeHTML(s)}</button>`).join("")}
                   </div>
                 </div>`
              : chatMessages
                  .map((m) =>
                    m.role === "assistant"
                      ? `<div class="chat-row assistant"><div class="chat-avatar" aria-hidden="true">TS</div><div class="chat-msg assistant md">${renderMarkdown(m.content)}</div></div>`
                      : `<div class="chat-row user"><div class="chat-msg user">${escapeHTML(m.content)}</div></div>`
                  )
                  .join("")
          }
          ${chatLoading ? `<div class="chat-row assistant"><div class="chat-avatar" aria-hidden="true">TS</div><div class="chat-msg assistant" role="status" aria-label="Assistant is typing"><span class="typing-dots"><i></i><i></i><i></i></span></div></div>` : ""}
        </div>
        <div class="chat-input-row">
          <label class="sr-only" for="chat-input">Ask about this engine</label>
          <input type="text" id="chat-input" placeholder="Ask about risk, sensors, SHAP" autocomplete="off" />
          <button class="send-btn" id="chat-send" aria-label="Send message">${ICONS.send}</button>
        </div>
      </div>
    `;
    scrollChatToBottom();
    const chatRows = slot.querySelectorAll(".chat-row");
    if (chatRows.length) Fx.rise([chatRows[chatRows.length - 1]], { y: 10, duration: 0.4 });
    const input = document.getElementById("chat-input");
    slot.querySelectorAll(".chat-chip").forEach((chip) =>
      chip.addEventListener("click", () => {
        input.value = chip.dataset.q;
        send();
      })
    );
    const send = async () => {
      const question = input.value.trim();
      if (!question || chatLoading) return;
      input.value = "";
      chatMessages.push({ role: "user", content: question });
      chatLoading = true;
      renderChatSlot();
      document.getElementById("chat-input").focus();

      // The reply is written into one bubble as it streams in, so the rest of
      // the chat (and anything the visitor is typing) is left alone.
      let bubble = null;
      let target = ""; // everything received from the server so far
      let shown = 0; // how much of it has been revealed
      let streamDone = false;

      // Close half-typed bold/code so "**Limitations" never shows raw asterisks.
      const closeOpenMarkup = (s) => {
        if ((s.match(/\*\*/g) || []).length % 2) s += "**";
        if ((s.match(/`/g) || []).length % 2) s += "`";
        return s;
      };

      const paint = (text) => {
        const body = document.getElementById("chat-body");
        if (!body) return; // visitor navigated away mid-answer
        if (!bubble || !bubble.isConnected) {
          body.querySelector(".typing-dots")?.closest(".chat-row")?.remove();
          body.insertAdjacentHTML(
            "beforeend",
            `<div class="chat-row assistant"><div class="chat-avatar" aria-hidden="true">TS</div><div class="chat-msg assistant md streaming"></div></div>`
          );
          bubble = body.lastElementChild.querySelector(".chat-msg");
        }
        const nearBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 90;
        bubble.innerHTML = renderMarkdown(closeOpenMarkup(text));
        if (nearBottom) body.scrollTop = body.scrollHeight;
      };

      // The server (especially the language model) can send text far faster
      // than is comfortable to read, so reveal it at a steady pace and speed
      // up gently only if the reveal falls well behind what has arrived.
      const TICK_MS = 30;
      const BASE_CHARS_PER_SEC = 140;
      let finishTyping;
      const typed = new Promise((resolve) => (finishTyping = resolve));
      const timer = setInterval(() => {
        const backlog = target.length - shown;
        if (backlog > 0) {
          const step = Math.max(2, Math.ceil((BASE_CHARS_PER_SEC * TICK_MS) / 1000 + backlog * 0.004));
          shown = Math.min(target.length, shown + step);
          paint(target.slice(0, shown));
        }
        if (streamDone && shown >= target.length) {
          clearInterval(timer);
          finishTyping();
        }
      }, TICK_MS);

      try {
        await API.chatStream(engine.unit, question, subset, analysisRevealed, (text) => (target = text));
        streamDone = true;
        await typed;
        chatMessages.push({ role: "assistant", content: target });
        bubble?.classList.remove("streaming");
        chatLoading = false;
      } catch {
        clearInterval(timer);
        chatLoading = false;
        if (target) {
          // Keep what arrived rather than throwing it away.
          paint(target);
          chatMessages.push({ role: "assistant", content: target });
          bubble?.classList.remove("streaming");
          toast("The answer was cut short.", "error");
        } else {
          toast("Couldn't reach the chat backend.", "error");
          chatMessages.pop();
          renderChatSlot();
        }
      }
    };
    document.getElementById("chat-send").addEventListener("click", send);
    input.addEventListener("keydown", (e) => e.key === "Enter" && send());
  }

  function scrollChatToBottom() {
    // rAF, not a synchronous read right after innerHTML — makes sure layout
    // (including the very first message replacing the empty-state markup)
    // has actually been committed before we measure scrollHeight.
    requestAnimationFrame(() => {
      const body = document.getElementById("chat-body");
      if (body) body.scrollTop = body.scrollHeight;
    });
  }

  function escapeHTML(str) {
    const div = document.createElement("div");
    div.textContent = str ?? "";
    return div.innerHTML;
  }

  const CHAT_SUGGESTIONS = [
    "Why is this engine at this risk level?",
    "Which sensors matter most?",
    "How many cycles are left?",
    "What does the critique agent think?",
  ];

  // Small, safe markdown renderer: escapes everything first, then turns
  // **bold**, *italic*, `code`, bullet/numbered lists and paragraphs into HTML.
  function renderMarkdown(src) {
    const inline = (t) =>
      escapeHTML(t)
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
    const lines = String(src ?? "").replace(/\r/g, "").split("\n");
    let html = "";
    let list = null; // "ul" | "ol"
    const close = () => { if (list) { html += `</${list}>`; list = null; } };
    for (const raw of lines) {
      const line = raw.trim();
      let m;
      if (!line) { close(); continue; }
      if ((m = line.match(/^[-*•]\s+(.*)/))) {
        if (list !== "ul") { close(); html += "<ul>"; list = "ul"; }
        html += `<li>${inline(m[1])}</li>`;
      } else if ((m = line.match(/^\d+[.)]\s+(.*)/))) {
        if (list !== "ol") { close(); html += "<ol>"; list = "ol"; }
        html += `<li>${inline(m[1])}</li>`;
      } else if ((m = line.match(/^#{1,4}\s+(.*)/))) {
        close();
        html += `<p class="md-h">${inline(m[1])}</p>`;
      } else {
        close();
        html += `<p>${inline(line)}</p>`;
      }
    }
    close();
    return html;
  }

  function resetChat() {
    chatMessages = [];
    chatLoading = false;
  }

  return { render, resetChat };
})();
