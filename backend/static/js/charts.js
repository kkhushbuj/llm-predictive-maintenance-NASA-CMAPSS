const Charts = (() => {
  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  let warnedMissingChart = false;
  function chartLibAvailable() {
    if (typeof Chart !== "undefined") return true;
    if (!warnedMissingChart) {
      console.warn("Chart.js failed to load (CDN unreachable?) — charts will be skipped.");
      warnedMissingChart = true;
    }
    return false;
  }

  const registry = new Map();

  function destroy(canvasId) {
    const existing = registry.get(canvasId);
    if (existing) {
      existing.destroy();
      registry.delete(canvasId);
    }
  }

  function baseGrid() {
    return { color: cssVar("--border"), drawTicks: false };
  }
  function baseTicks() {
    return { color: cssVar("--text-dim"), font: { size: 12, family: "Inter, system-ui, sans-serif" } };
  }
  function axisTitle(text) {
    return { display: true, text, color: cssVar("--text-dim"), font: { size: 12, family: "Inter, system-ui, sans-serif" } };
  }
  function tooltipStyle() {
    return {
      backgroundColor: cssVar("--bg-elevated"),
      titleColor: cssVar("--text"),
      bodyColor: cssVar("--text-dim"),
      borderColor: cssVar("--border-strong"),
      borderWidth: 1,
      padding: 10,
      cornerRadius: 8,
      displayColors: false,
      titleFont: { family: "Inter, system-ui, sans-serif", weight: "600", size: 13 },
      bodyFont: { family: "Inter, system-ui, sans-serif", size: 12.5 },
    };
  }

  // `scale` converts stored SHAP values to cycles (FD002/FD004 store them as
  // fractions of the 125-cycle RUL cap).
  function shapChart(canvasId, sensors, scale = 1) {
    if (!chartLibAvailable()) return null;
    destroy(canvasId);
    const sorted = [...sensors].sort((a, b) => a.importance - b.importance);
    const labels = sorted.map((s) => s.sensor.replace("sensor_", "S"));
    const values = sorted.map((s) => s.signed_contribution * scale);
    const colors = sorted.map((s) =>
      s.direction === "increases_risk" ? cssVar("--risk-high") : cssVar("--risk-low")
    );

    const ctx = document.getElementById(canvasId).getContext("2d");
    const chart = new Chart(ctx, {
      type: "bar",
      data: { labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 4, barThickness: 18 }] },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            ...tooltipStyle(),
            callbacks: {
              title: (items) => `Sensor ${sorted[items[0].dataIndex].sensor.replace("sensor_", "")}`,
              label: (item) => {
                const s = sorted[item.dataIndex];
                const dir =
                  s.direction === "increases_risk"
                    ? "Shortens remaining life → raises risk"
                    : "Lengthens remaining life → lowers risk";
                return `${Number(item.raw).toFixed(2)} cycles · ${dir}`;
              },
            },
          },
        },
        scales: {
          x: {
            title: axisTitle("Effect on predicted remaining life (cycles)"),
            grid: baseGrid(),
            border: { display: false },
            ticks: { ...baseTicks(), padding: 4 },
          },
          y: { grid: { display: false }, border: { display: false }, ticks: { color: cssVar("--text"), font: { size: 12.5, weight: 600, family: "Inter, system-ui, sans-serif" } } },
        },
      },
    });
    registry.set(canvasId, chart);
    return chart;
  }

  function sensorTrendChart(canvasId, rows, sensorKey) {
    if (!chartLibAvailable()) return null;
    destroy(canvasId);
    const labels = rows.map((r) => r.cycle);
    const values = rows.map((r) => r[sensorKey]);

    const ctx = document.getElementById(canvasId).getContext("2d");
    const gradient = ctx.createLinearGradient(0, 0, 0, 220);
    const accent = cssVar("--accent");
    gradient.addColorStop(0, accent + "26");
    gradient.addColorStop(1, accent + "00");

    const chart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            data: values,
            borderColor: cssVar("--accent"),
            backgroundColor: gradient,
            fill: true,
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 5,
            pointHoverBackgroundColor: cssVar("--accent"),
            pointHoverBorderColor: cssVar("--bg-elevated"),
            pointHoverBorderWidth: 2,
            tension: 0.25,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            ...tooltipStyle(),
            callbacks: {
              title: (items) => `Cycle ${items[0].label}`,
              label: (item) => `Normalized value ${Number(item.raw).toFixed(3)}`,
            },
          },
        },
        scales: {
          x: {
            title: axisTitle("Cycle"),
            grid: { display: false },
            border: { color: cssVar("--border-strong") },
            ticks: { ...baseTicks(), maxTicksLimit: 10 },
          },
          y: { grid: baseGrid(), border: { display: false }, ticks: { ...baseTicks(), padding: 4 } },
        },
      },
    });
    registry.set(canvasId, chart);
    return chart;
  }

  function calibrationChart(canvasId, engines, onSelect) {
    if (!chartLibAvailable()) return null;
    destroy(canvasId);
    const tiers = { low: [], medium: [], high: [] };
    let maxRul = 0;
    for (const e of engines) {
      tiers[riskTier(e.risk_score)].push({ x: e.true_RUL, y: e.predicted_RUL, unit: e.unit });
      maxRul = Math.max(maxRul, e.true_RUL, e.predicted_RUL);
    }
    maxRul = Math.ceil((maxRul + 5) / 10) * 10;

    const surface = cssVar("--card");
    const dot = (color) => ({
      backgroundColor: color + "d9",
      borderColor: surface,
      borderWidth: 1.5,
      pointRadius: 4.5,
      pointHoverRadius: 7,
      hitRadius: 14,
    });

    const ctx = document.getElementById(canvasId).getContext("2d");
    const chart = new Chart(ctx, {
      type: "scatter",
      data: {
        datasets: [
          { label: "High risk", data: tiers.high, ...dot(cssVar("--risk-high")) },
          { label: "Medium risk", data: tiers.medium, ...dot(cssVar("--risk-medium")) },
          { label: "Low risk", data: tiers.low, ...dot(cssVar("--risk-low")) },
          {
            label: "Perfect prediction",
            data: [{ x: 0, y: 0 }, { x: maxRul, y: maxRul }],
            type: "line",
            borderColor: cssVar("--text-faint"),
            borderWidth: 1,
            pointRadius: 0,
            pointHitRadius: 0,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "nearest", intersect: true },
        animation: {
          duration: 650,
          easing: "easeOutQuart",
          delay: (ctx) => (ctx.type === "data" && ctx.datasetIndex < 3 ? 300 + ctx.dataIndex * 10 : 0),
        },
        onHover: (evt, active) => {
          evt.native.target.style.cursor = active.some((a) => a.datasetIndex < 3) ? "pointer" : "default";
        },
        onClick: (evt, active) => {
          const hit = active.find((a) => a.datasetIndex < 3);
          if (hit && onSelect) onSelect(chart.data.datasets[hit.datasetIndex].data[hit.index].unit);
        },
        plugins: {
          legend: {
            display: true,
            position: "bottom",
            labels: {
              color: cssVar("--text-dim"),
              usePointStyle: true,
              pointStyle: "circle",
              boxWidth: 8,
              boxHeight: 8,
              padding: 14,
              font: { size: 12, family: "Inter, system-ui, sans-serif" },
              filter: (item) => item.datasetIndex < 3,
            },
          },
          tooltip: {
            ...tooltipStyle(),
            filter: (item) => item.datasetIndex < 3,
            callbacks: {
              title: (items) => `Engine ${items[0].raw.unit}`,
              label: (item) => {
                const err = item.raw.y - item.raw.x;
                return [`Predicted ${item.raw.y.toFixed(1)} · true ${item.raw.x.toFixed(0)} cycles`, `Error ${err >= 0 ? "+" : "−"}${Math.abs(err).toFixed(1)} · ${item.dataset.label.toLowerCase()}`];
              },
            },
          },
        },
        scales: {
          x: { min: 0, max: maxRul, title: axisTitle("True RUL (cycles)"), grid: baseGrid(), border: { display: false }, ticks: baseTicks() },
          y: { min: 0, max: maxRul, title: axisTitle("Predicted RUL (cycles)"), grid: baseGrid(), border: { display: false }, ticks: baseTicks() },
        },
      },
    });
    registry.set(canvasId, chart);
    return chart;
  }

  function gaugeChart(canvasId, score) {
    if (!chartLibAvailable()) return null;
    destroy(canvasId);
    const tier = riskTier(score);
    const color = cssVar(`--risk-${tier}`);
    const ctx = document.getElementById(canvasId).getContext("2d");
    const chart = new Chart(ctx, {
      type: "doughnut",
      data: {
        datasets: [
          {
            data: [score, 100 - score],
            backgroundColor: [color, cssVar("--border")],
            borderWidth: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "80%",
        circumference: 360,
        rotation: -90,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        animation: { animateRotate: true, duration: 600 },
      },
    });
    registry.set(canvasId, chart);
    return chart;
  }

  function riskTier(score) {
    if (score >= 70) return "high";
    if (score >= 40) return "medium";
    return "low";
  }

  return { shapChart, sensorTrendChart, calibrationChart, gaugeChart, riskTier, destroy };
})();
