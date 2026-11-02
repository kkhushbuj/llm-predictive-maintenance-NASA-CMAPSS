const Dashboard = (() => {
  let engines = [];
  let subset = "FD001";
  let sortKey = "risk_rank";
  let sortDir = "asc";
  let query = "";
  let onlyDeep = false;

  const SUBSET_LABELS = {
    FD001: "single operating condition, HPC degradation",
    FD002: "six operating conditions, HPC degradation",
    FD004: "six operating conditions, HPC + fan degradation",
  };

  const ICON_SPARK =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/></svg>';

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function riskBadge(score) {
    const tier = Charts.riskTier(score);
    const label = tier === "high" ? "High risk" : tier === "medium" ? "Medium risk" : "Low risk";
    return `<span class="risk-badge ${tier}"><span class="rb-dot"></span>${label}</span>`;
  }

  function subsetSwitcherHTML(subsets) {
    const names = subsets.length ? subsets.map((s) => s.subset) : SUBSETS;
    const buttons = names
      .map((s) => `<button role="tab" aria-selected="${s === subset}" data-subset="${s}">${s}</button>`)
      .join("");
    return `<div class="seg" role="tablist" aria-label="C-MAPSS dataset" id="subset-switcher"><span class="seg-ind" aria-hidden="true"></span>${buttons}</div>`;
  }

  // The selector's position is remembered across renders so the marker
  // slides from the previous dataset to the new one.
  let lastSeg = null;

  function placeSegIndicator(animate) {
    const ind = document.querySelector("#subset-switcher .seg-ind");
    const sel = document.querySelector('#subset-switcher button[aria-selected="true"]');
    if (!ind || !sel) return;
    const to = { left: sel.offsetLeft, width: sel.offsetWidth };
    Fx.slide(ind, animate ? lastSeg : to, to);
    lastSeg = to;
  }

  function bindSwitcher() {
    document.querySelectorAll("#subset-switcher [data-subset]").forEach((btn) => {
      btn.addEventListener("click", () => Router.goSubset(btn.dataset.subset));
    });
  }

  function kpiPlaceholders() {
    return ["Test RMSE", "NASA score", "Confirmed by literature", "Contradicted or novel"]
      .map(
        (label) => `
        <div class="kpi">
          <p class="kpi-label">${label}</p>
          <p class="kpi-value"><span class="skeleton" style="display:inline-block;width:72px;height:28px;"></span></p>
          <p class="kpi-hint">&nbsp;</p>
        </div>`
      )
      .join("");
  }

  async function render(initialSubset) {
    subset = initialSubset || "FD001";
    const app = document.getElementById("app");
    app.innerHTML = `
      <header class="hero page-head">
        <div class="hero-bg" id="hero-bg" aria-hidden="true"></div>
        <div class="hero-inner">
          <div>
            <h1>Fleet overview</h1>
            <p class="page-sub">LSTM remaining-useful-life predictions for the NASA C-MAPSS <strong>${subset}</strong> fleet (${SUBSET_LABELS[subset] || ""}), explained with SHAP and cross-checked against published literature by a multi-agent pipeline.</p>
            <div class="hero-chips" id="hero-chips" aria-label="Fleet summary"></div>
          </div>
          <div id="subset-switcher-slot">${subsetSwitcherHTML([])}</div>
        </div>
      </header>

      <div class="ticker" id="ticker" aria-label="Highest-risk engines">
        <span class="ticker-label"><i class="live-dot" aria-hidden="true"></i>Fleet watch</span>
        <div class="ticker-track"><div class="ticker-row" id="ticker-row"></div></div>
      </div>

      <section class="kpis" id="kpi-grid" aria-label="Model summary">${kpiPlaceholders()}</section>

      <section class="two-col">
        <div class="panel fleet-panel">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">Fleet health</h2>
              <p class="panel-sub" id="fleet-sub">Engines in unit order. Select one to open its detail page.</p>
            </div>
            <div class="legend" aria-label="Legend">
              <span class="legend-item"><span class="legend-swatch" style="background:var(--risk-low)"></span>Low &lt;40</span>
              <span class="legend-item"><span class="legend-swatch" style="background:var(--risk-medium)"></span>Medium 40–69</span>
              <span class="legend-item"><span class="legend-swatch" style="background:var(--risk-high)"></span>High 70+</span>
              <span class="legend-item"><span class="legend-swatch ring"></span>Deep-analyzed</span>
            </div>
          </div>
          <div class="fleet-grid" id="fleet-grid" role="group" aria-label="Engine fleet, colored by risk"></div>
          <div class="fleet-extra" id="fleet-extra"></div>
        </div>
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">Predicted vs. true RUL</h2>
              <p class="panel-sub" id="calibration-sub">Every test engine. Points on the diagonal are perfect predictions.</p>
            </div>
          </div>
          <div class="chart-box fill"><canvas id="calibration-chart" role="img" aria-label="Scatter chart of predicted against true remaining useful life for every test engine"></canvas></div>
        </div>
      </section>

      <section class="panel" aria-label="Engine list">
        <div class="table-controls">
          <label class="sr-only" for="engine-search">Search by engine unit</label>
          <input class="search-input" id="engine-search" type="search" inputmode="numeric" placeholder="Search by engine unit" autocomplete="off" />
          <button class="filter-chip" id="deep-filter" aria-pressed="false">${ICON_SPARK.replace("<svg", '<svg width="13" height="13"')} Deep-analyzed only</button>
          <span class="table-count" id="table-count" aria-live="polite"></span>
        </div>
        <div class="table-wrap">
          <table>
            <caption class="sr-only">Test engines ranked by risk. Column headers sort the table.</caption>
            <thead>
              <tr>
                <th scope="col" data-sort="risk_rank" aria-sort="ascending"><button class="th-btn" type="button">Rank<span class="sort-ind" aria-hidden="true">▲</span></button></th>
                <th scope="col" data-sort="unit"><button class="th-btn" type="button">Engine<span class="sort-ind" aria-hidden="true">▲</span></button></th>
                <th scope="col" data-sort="risk_score"><button class="th-btn" type="button">Risk<span class="sort-ind" aria-hidden="true">▲</span></button></th>
                <th scope="col" class="num" data-sort="predicted_RUL"><button class="th-btn" type="button">Predicted RUL<span class="sort-ind" aria-hidden="true">▲</span></button></th>
                <th scope="col" class="num" data-sort="true_RUL"><button class="th-btn" type="button">True RUL<span class="sort-ind" aria-hidden="true">▲</span></button></th>
                <th scope="col" class="plain num">Analysis</th>
              </tr>
            </thead>
            <tbody id="engine-tbody"></tbody>
          </table>
        </div>
      </section>
    `;

    Fx.rise(".page-head, .kpi", { gap: 0.07 });
    Fx.words(".hero h1");
    Fx.background("#hero-bg");
    bindSwitcher();
    placeSegIndicator(true);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => placeSegIndicator(false));
    Fx.rise(".two-col .panel", { delay: 0.3, gap: 0.1 });
    Fx.rise('section[aria-label="Engine list"]', { delay: 0.45 });

    const search = document.getElementById("engine-search");
    query = "";
    onlyDeep = false;
    sortKey = "risk_rank";
    sortDir = "asc";
    search.addEventListener("input", (e) => {
      query = e.target.value.trim();
      renderTable();
    });
    document.getElementById("deep-filter").addEventListener("click", (e) => {
      onlyDeep = !onlyDeep;
      e.currentTarget.setAttribute("aria-pressed", String(onlyDeep));
      renderTable();
    });
    document.querySelectorAll("th[data-sort]").forEach((th) => {
      th.querySelector("button").addEventListener("click", () => {
        if (sortKey === th.dataset.sort) sortDir = sortDir === "asc" ? "desc" : "asc";
        else {
          sortKey = th.dataset.sort;
          sortDir = th.dataset.sort === "risk_score" ? "desc" : "asc";
        }
        renderTable();
      });
    });

    const [subsets, metrics, summary, engineList] = await Promise.all([
      API.subsets().catch(() => []),
      API.modelMetrics(subset).catch(() => null),
      API.agentSummary(subset).catch(() => null),
      API.engines(subset).catch(() => null),
    ]);

    // Only rebuild the selector when the available datasets differ from the
    // default list, so the sliding marker isn't interrupted.
    const names = subsets.map((s) => s.subset);
    if (names.length && names.join() !== SUBSETS.join()) {
      document.getElementById("subset-switcher-slot").innerHTML = subsetSwitcherHTML(subsets);
      bindSwitcher();
      placeSegIndicator(false);
    }

    if (!engineList || !engineList.length) {
      renderLoadError();
      return;
    }

    engines = engineList;
    document.getElementById("fleet-sub").textContent =
      `All ${engines.length} engines in unit order. Select one to open its detail page.`;
    renderStats(metrics, summary);
    renderFleetGrid();
    renderFleetExtra();
    renderHeroChips();
    renderTicker();
    Charts.calibrationChart("calibration-chart", engines, (unit) => Router.goEngine(unit));
    renderCalibrationSummary();
    renderTable(true);
  }

  function renderLoadError() {
    document.getElementById("kpi-grid").innerHTML = "";
    document.getElementById("ticker")?.remove();
    document.querySelector(".two-col").outerHTML = "";
    document.querySelector('section[aria-label="Engine list"]').outerHTML = `
      <div class="state-block error" role="alert">
        <h2>Couldn't load the ${subset} fleet</h2>
        <p>The API didn't return any engines for this dataset. Check that the backend is running and that results exist for ${subset}, then try again.</p>
        <button class="btn" id="retry-load">Retry</button>
      </div>`;
    document.getElementById("retry-load").addEventListener("click", () => Router.refresh());
  }

  function renderStats(metrics, summary) {
    const cards = document.querySelectorAll("#kpi-grid .kpi");
    const set = (card, value, decimals, hint) => {
      const el = card.querySelector(".kpi-value");
      if (value == null) el.textContent = "—";
      else Fx.count(el, value, decimals, { delay: 0.1 * [...cards].indexOf(card) });
      card.querySelector(".kpi-hint").textContent = hint;
    };
    set(cards[0], metrics?.test_rmse, 2, `Cycles, ${subset} test set`);
    set(cards[1], metrics?.test_nasa_score, 0, "Asymmetric PHM08 metric, lower is better");
    set(cards[2], summary?.confirmed, 0, `Of ${summary?.total_engines_analyzed ?? 0} deep-analyzed engines`);
    set(
      cards[3],
      summary?.contradicted != null ? summary.contradicted + summary.novel : null,
      0,
      `${summary?.contradicted ?? 0} contradicted, ${summary?.novel ?? 0} novel`
    );
  }

  function renderCalibrationSummary() {
    const errs = engines.map((e) => Math.abs(e.predicted_RUL - e.true_RUL)).sort((a, b) => a - b);
    const median = errs[Math.floor(errs.length / 2)];
    const within = errs.filter((x) => x <= 10).length;
    const text = `Median absolute error is ${median.toFixed(1)} cycles, and ${within} of ${errs.length} engines are within 10 cycles of the true RUL.`;
    document.getElementById("calibration-sub").textContent = text;
    document.getElementById("calibration-chart").setAttribute("aria-label", `Scatter chart of predicted against true remaining useful life. ${text}`);
  }

  function renderFleetGrid() {
    const grid = document.getElementById("fleet-grid");
    const byUnit = [...engines].sort((a, b) => a.unit - b.unit);
    const tooltip = ensureTooltip();

    grid.innerHTML = byUnit
      .map((e) => {
        const tier = Charts.riskTier(e.risk_score);
        const tierLabel = tier === "high" ? "high risk" : tier === "medium" ? "medium risk" : "low risk";
        const label = `Engine ${e.unit}, ${tierLabel}, predicted RUL ${e.predicted_RUL.toFixed(1)} cycles${e.deep_analysis ? ", deep-analyzed" : ""}`;
        return `<button class="fleet-cell ${tier} ${e.deep_analysis ? "deep" : ""}" data-unit="${e.unit}" data-rul="${e.predicted_RUL.toFixed(1)}" data-risk="${e.risk_score.toFixed(0)}" data-tier="${tierLabel}" aria-label="${label}"></button>`;
      })
      .join("");

    Fx.wave(grid.querySelectorAll(".fleet-cell"), { columns: window.innerWidth > 700 ? 20 : 10 });
    enableDockEffect(grid);

    const show = (cell, x, y) => {
      tooltip.style.display = "block";
      tooltip.style.left = Math.min(x + 14, window.innerWidth - tooltip.offsetWidth - 8) + "px";
      tooltip.style.top = y + 14 + "px";
      tooltip.innerHTML = `<strong>Engine ${cell.dataset.unit}</strong><br/>${cell.dataset.rul} cycles left · risk ${cell.dataset.risk} (${cell.dataset.tier})`;
    };
    grid.querySelectorAll(".fleet-cell").forEach((cell) => {
      cell.addEventListener("click", () => Router.goEngine(cell.dataset.unit));
      cell.addEventListener("pointermove", (ev) => show(cell, ev.clientX, ev.clientY));
      cell.addEventListener("pointerleave", () => (tooltip.style.display = "none"));
      cell.addEventListener("focus", () => {
        const r = cell.getBoundingClientRect();
        show(cell, r.left, r.bottom);
      });
      cell.addEventListener("blur", () => (tooltip.style.display = "none"));
    });
  }

  function renderHeroChips() {
    const high = engines.filter((e) => Charts.riskTier(e.risk_score) === "high").length;
    const deep = engines.filter((e) => e.deep_analysis).length;
    const errs = engines.map((e) => Math.abs(e.predicted_RUL - e.true_RUL)).sort((a, b) => a - b);
    const median = errs[Math.floor(errs.length / 2)];
    const chips = [
      { n: engines.length, d: 0, label: "engines" },
      { n: high, d: 0, label: "high risk", cls: "high" },
      { n: deep, d: 0, label: "deep-analyzed" },
      { n: median, d: 1, label: "cycles median error" },
    ];
    const host = document.getElementById("hero-chips");
    host.innerHTML = chips
      .map((c) => `<span class="hchip ${c.cls || ""}">${c.cls ? '<i class="hdot" aria-hidden="true"></i>' : ""}<b>${c.n.toFixed(c.d)}</b> ${c.label}</span>`)
      .join("");
    host.querySelectorAll("b").forEach((el, i) => Fx.count(el, chips[i].n, chips[i].d, { delay: 0.15 * i, duration: 0.9 }));
    Fx.rise(host.querySelectorAll(".hchip"), { gap: 0.08, y: 10, duration: 0.5 });
  }

  function renderTicker() {
    const top = [...engines].sort((a, b) => a.risk_rank - b.risk_rank).slice(0, 12);
    const item = (e, hidden) => {
      const tier = Charts.riskTier(e.risk_score);
      const label = tier === "high" ? "High" : tier === "medium" ? "Medium" : "Low";
      return `<a class="tick ${tier}" href="${Router.engineHref(e.unit, subset)}"${hidden ? ' tabindex="-1" aria-hidden="true"' : ""}>
        <b>Engine ${e.unit}</b><span>${e.predicted_RUL.toFixed(1)} cycles left</span><em>${label}</em></a>`;
    };
    const set = top.map((e) => item(e, false)).join("");
    const copy = top.map((e) => item(e, true)).join("");
    document.getElementById("ticker-row").innerHTML = set + copy;
    Fx.rise("#ticker", { y: 8, duration: 0.5 });
  }

  function renderFleetExtra() {
    const counts = { low: 0, medium: 0, high: 0 };
    engines.forEach((e) => counts[Charts.riskTier(e.risk_score)]++);
    const total = engines.length || 1;
    const top = [...engines].sort((a, b) => a.risk_rank - b.risk_rank).slice(0, 4);

    const seg = (tier) =>
      counts[tier] ? `<span style="width:${(counts[tier] / total) * 100}%;background:var(--risk-${tier})" title="${counts[tier]} ${tier} risk"></span>` : "";

    document.getElementById("fleet-extra").innerHTML = `
      <h3>Risk split</h3>
      <div class="risk-split" role="img" aria-label="${counts.low} low, ${counts.medium} medium and ${counts.high} high risk engines">${seg("low")}${seg("medium")}${seg("high")}</div>
      <div class="split-legend">
        <span><b>${counts.low}</b> low</span>
        <span><b>${counts.medium}</b> medium</span>
        <span><b>${counts.high}</b> high</span>
      </div>
      <h3 class="top-title">Highest-risk engines</h3>
      <ul class="top-list">
        ${top
          .map(
            (e) => `
          <li><a href="${Router.engineHref(e.unit, subset)}">
            <span class="top-rank">#${e.risk_rank}</span>
            <span class="top-name">Engine ${e.unit}</span>
            <span class="top-rul">${e.predicted_RUL.toFixed(1)} cycles left</span>
            ${riskBadge(e.risk_score)}
          </a></li>`
          )
          .join("")}
      </ul>`;
    Fx.rise("#fleet-extra .top-list li", { delay: 0.5, gap: 0.07, y: 8, duration: 0.45 });
  }

  // Tiles near the pointer swell like a dock; CSS reads --s for the scale.
  function enableDockEffect(grid) {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const cells = [...grid.querySelectorAll(".fleet-cell")];
    const RADIUS = 78;
    const MAX = 0.55;
    let raf = null;
    let last = null;

    const apply = () => {
      raf = null;
      if (!last) return;
      const scales = cells.map((c) => {
        const r = c.getBoundingClientRect();
        const d = Math.hypot(last.x - (r.left + r.width / 2), last.y - (r.top + r.height / 2));
        const t = Math.max(0, 1 - d / RADIUS);
        return 1 + MAX * t * t;
      });
      cells.forEach((c, i) => {
        c.style.setProperty("--s", scales[i].toFixed(3));
        c.style.zIndex = scales[i] > 1.02 ? String(Math.round(scales[i] * 10)) : "";
      });
    };

    grid.addEventListener("pointermove", (e) => {
      if (e.pointerType !== "mouse") return;
      last = { x: e.clientX, y: e.clientY };
      if (!raf) raf = requestAnimationFrame(apply);
    });
    grid.addEventListener("pointerleave", () => {
      last = null;
      cells.forEach((c) => {
        c.style.removeProperty("--s");
        c.style.zIndex = "";
      });
    });
  }

  function ensureTooltip() {
    let tooltip = document.getElementById("cell-tooltip");
    if (!tooltip) {
      tooltip = document.createElement("div");
      tooltip.id = "cell-tooltip";
      tooltip.className = "cell-tooltip";
      tooltip.setAttribute("aria-hidden", "true");
      document.body.appendChild(tooltip);
    }
    return tooltip;
  }

  function renderTable(animate = false) {
    let rows = engines;
    if (query) rows = rows.filter((e) => String(e.unit).includes(query));
    if (onlyDeep) rows = rows.filter((e) => e.deep_analysis);
    const dir = sortDir === "asc" ? 1 : -1;
    rows = [...rows].sort((a, b) => (a[sortKey] - b[sortKey]) * dir);

    document.querySelectorAll("th[data-sort]").forEach((th) => {
      if (th.dataset.sort === sortKey) th.setAttribute("aria-sort", sortDir === "asc" ? "ascending" : "descending");
      else th.removeAttribute("aria-sort");
      th.querySelector(".sort-ind").textContent = th.dataset.sort === sortKey && sortDir === "desc" ? "▼" : "▲";
    });

    document.getElementById("table-count").textContent = `${rows.length} of ${engines.length} engines`;
    const tbody = document.getElementById("engine-tbody");

    // Remember where visible rows were so they can glide to their new spots.
    const before = new Map();
    if (!animate && Fx.on()) {
      [...tbody.querySelectorAll("tr[data-unit]")].slice(0, 60).forEach((tr) => before.set(tr.dataset.unit, tr.getBoundingClientRect().top));
    }

    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="6" class="table-empty">No engines match${query ? ` “${esc(query)}”` : " these filters"}.<br/><button class="btn secondary" id="clear-filters">Clear filters</button></td></tr>`;
      document.getElementById("clear-filters").addEventListener("click", (e) => {
        e.stopPropagation();
        query = "";
        onlyDeep = false;
        document.getElementById("engine-search").value = "";
        document.getElementById("deep-filter").setAttribute("aria-pressed", "false");
        renderTable();
      });
      return;
    }

    tbody.innerHTML = rows
      .map((e) => {
        const tier = Charts.riskTier(e.risk_score);
        return `
        <tr data-unit="${e.unit}">
          <td class="num text-dim" style="text-align:left">#${e.risk_rank}</td>
          <td><a class="engine-link" href="${Router.engineHref(e.unit, subset)}">Engine ${e.unit}</a></td>
          <td>
            <span class="risk-bar" aria-hidden="true"><span class="risk-bar-fill" style="width:${e.risk_score}%;background:var(--risk-${tier})"></span></span>
            ${riskBadge(e.risk_score)}
          </td>
          <td class="num">${e.predicted_RUL.toFixed(1)} cycles</td>
          <td class="num text-dim">${e.true_RUL.toFixed(0)} cycles</td>
          <td class="num">${
            e.deep_analysis
              ? `<span class="badge-deep">${ICON_SPARK} Analyzed</span>`
              : `<a class="badge-deep-outline" href="${Router.engineHref(e.unit, subset)}">Run analysis</a>`
          }</td>
        </tr>`;
      })
      .join("");

    tbody.querySelectorAll("tr[data-unit]").forEach((tr) => {
      tr.addEventListener("click", (ev) => {
        if (ev.target.closest("a")) return;
        Router.goEngine(tr.dataset.unit);
      });
    });
    if (before.size) {
      const glide = { duration: 420, easing: "cubic-bezier(0.22, 1, 0.36, 1)" };
      [...tbody.querySelectorAll("tr[data-unit]")].slice(0, 60).forEach((tr) => {
        const prev = before.get(tr.dataset.unit);
        if (prev === undefined) {
          tr.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 260, easing: "ease-out" });
          return;
        }
        const dy = prev - tr.getBoundingClientRect().top;
        if (Math.abs(dy) > 2) tr.animate([{ transform: `translateY(${dy}px)` }, { transform: "none" }], glide);
      });
    }
    if (animate) Fx.rise([...tbody.querySelectorAll("tr[data-unit]")].slice(0, 12), { gap: 0.03, y: 8, duration: 0.45, delay: 0.6 });
  }

  return { render };
})();
