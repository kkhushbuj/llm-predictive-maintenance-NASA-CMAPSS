function toast(message, type = "success") {
  const root = document.getElementById("toast-root");
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = message;
  root.appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

const SUBSETS = ["FD001", "FD002", "FD004"];

const Router = (() => {
  let currentSubset = "FD001";

  function parse() {
    const hash = location.hash.replace(/^#/, "") || "/";
    const engineMatch = hash.match(/^\/(?:(FD00[124])\/)?engine\/(\d+)$/);
    if (engineMatch) {
      return { view: "engine", subset: engineMatch[1] || "FD001", unit: Number(engineMatch[2]) };
    }
    const dashMatch = hash.match(/^\/(FD00[124])?$/);
    return { view: "dashboard", subset: (dashMatch && dashMatch[1]) || "FD001" };
  }

  function go(path) {
    location.hash = path;
  }

  function engineHref(unit, subset = currentSubset) {
    return subset === "FD001" ? `#/engine/${unit}` : `#/${subset}/engine/${unit}`;
  }

  function goEngine(unit) {
    go(engineHref(unit).slice(1));
  }

  function goSubset(subset) {
    go(subset === "FD001" ? "/" : `/${subset}`);
  }

  async function render({ keepChat = false, resetScroll = true } = {}) {
    const route = parse();
    currentSubset = route.subset;
    Fx.destroyBackground();
    if (resetScroll) window.scrollTo({ top: 0, behavior: "instant" });
    if (route.view === "engine") {
      if (!keepChat) EngineView.resetChat();
      await EngineView.render(route.unit, route.subset);
    } else {
      await Dashboard.render(route.subset);
    }
    if (resetScroll) document.getElementById("app").focus({ preventScroll: true });
  }

  function init() {
    window.addEventListener("hashchange", () => render());
    render({ resetScroll: false });
  }

  return {
    go, goEngine, goSubset, engineHref, init,
    refresh: () => render({ keepChat: true, resetScroll: false }),
    get subset() { return currentSubset; },
  };
})();

const ICON_SUN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
const ICON_MOON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("turbinesense-theme"); } catch {}
  applyTheme(saved || "dark");
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    applyTheme(current === "dark" ? "light" : "dark");
    // Charts read theme colours at draw time, so redraw them.
    Router.refresh();
  });
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try { localStorage.setItem("turbinesense-theme", theme); } catch {}
  const btn = document.getElementById("theme-toggle");
  btn.innerHTML = theme === "dark" ? ICON_SUN : ICON_MOON;
  btn.setAttribute("aria-label", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
}

async function pollHealth() {
  const pill = document.getElementById("live-status");
  const text = pill.querySelector(".status-text");
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error();
    pill.classList.remove("offline");
    text.textContent = "API online";
  } catch {
    pill.classList.add("offline");
    text.textContent = "API offline";
  }
}

function initScrollProgress() {
  const bar = document.getElementById("scroll-progress");
  let ticking = false;
  const update = () => {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    bar.style.transform = `scaleX(${max > 0 ? Math.min(window.scrollY / max, 1) : 0})`;
    ticking = false;
  };
  window.addEventListener("scroll", () => {
    if (!ticking) {
      ticking = true;
      requestAnimationFrame(update);
    }
  }, { passive: true });
  window.addEventListener("resize", update);
  update();
}

initTheme();
initScrollProgress();
Fx.spotlight();
Router.init();
pollHealth();
setInterval(pollHealth, 20000);
