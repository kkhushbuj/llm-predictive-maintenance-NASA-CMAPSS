// Animation helpers built on Motion (CDN global `Motion`). Every helper is a
// no-op when Motion failed to load or the viewer prefers reduced motion, so
// content is never hidden by a script that doesn't run.
const Fx = (() => {
  const EASE = [0.22, 1, 0.36, 1];
  const M = () => window.Motion;
  const reduced = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const on = () => Boolean(M()) && !reduced();

  const list = (t) => (typeof t === "string" ? [...document.querySelectorAll(t)] : [...t].filter(Boolean));

  // Motion commits its final values as inline styles; clear them so CSS
  // hover/transform rules keep working afterwards.
  const clean = (anim, els) => {
    const clear = () =>
      els.forEach((el) => {
        el.style.opacity = "";
        el.style.transform = "";
      });
    Promise.resolve(anim).then(() => {
      clear();
      setTimeout(clear, 80);
    });
  };

  function rise(targets, { delay = 0, gap = 0.07, y = 18, duration = 0.6 } = {}) {
    const els = list(targets);
    if (!on() || !els.length) return;
    const anim = M().animate(
      els,
      { opacity: [0, 1], y: [y, 0] },
      { duration, ease: EASE, delay: M().stagger(gap, { startDelay: delay }) }
    );
    clean(anim, els);
  }

  // Diagonal wave across a grid of equally sized cells.
  function wave(targets, { columns, baseOpacity = 1, step = 0.018 } = {}) {
    const els = list(targets);
    if (!on() || !els.length) return;
    const anim = M().animate(
      els,
      { opacity: [0, baseOpacity], scale: [0.4, 1] },
      {
        duration: 0.5,
        ease: EASE,
        delay: (i) => 0.15 + ((i % columns) + Math.floor(i / columns)) * step,
      }
    );
    clean(anim, els);
  }

  function count(el, target, decimals = 0, { duration = 1.1, delay = 0 } = {}) {
    if (!on()) {
      el.textContent = target.toFixed(decimals);
      return;
    }
    el.textContent = (0).toFixed(decimals);
    M().animate(0, target, {
      duration,
      delay,
      ease: EASE,
      onUpdate: (v) => (el.textContent = v.toFixed(decimals)),
      onComplete: () => (el.textContent = target.toFixed(decimals)),
    });
  }

  // Vanta (three.js) hero background. Loaded lazily after first paint so it
  // never blocks rendering; any failure leaves the static CSS backdrop.
  let vantaLoad = null;
  let vantaFx = null;

  const loadScript = (src) =>
    new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src;
      s.async = true;
      s.onload = resolve;
      s.onerror = () => reject(new Error(`Failed to load ${src}`));
      document.head.appendChild(s);
    });

  function loadVanta() {
    if (!vantaLoad) {
      vantaLoad = loadScript("https://cdn.jsdelivr.net/npm/three@0.134.0/build/three.min.js")
        .then(() => loadScript("https://cdn.jsdelivr.net/npm/vanta@0.5.24/dist/vanta.net.min.js"))
        .catch((err) => {
          vantaLoad = null;
          throw err;
        });
    }
    return vantaLoad;
  }

  const hexToInt = (hex) => parseInt(hex.trim().replace("#", "").slice(0, 6), 16);
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function destroyBackground() {
    if (vantaFx) {
      try { vantaFx.destroy(); } catch {}
      vantaFx = null;
    }
  }

  async function background(selector) {
    destroyBackground();
    const el = document.querySelector(selector);
    if (!el || reduced() || window.innerWidth < 700) return;
    try {
      await loadVanta();
      // The view may have changed while the scripts were loading.
      const target = document.querySelector(selector);
      if (!target || !window.VANTA || !window.THREE) return;
      destroyBackground();
      vantaFx = window.VANTA.NET({
        el: target,
        THREE: window.THREE,
        mouseControls: true,
        touchControls: false,
        gyroControls: false,
        minHeight: 160,
        minWidth: 200,
        scale: 1,
        scaleMobile: 1,
        color: hexToInt(cssVar("--accent")),
        backgroundColor: hexToInt(cssVar("--card")),
        points: 8,
        maxDistance: 22,
        spacing: 19,
        showDots: true,
      });
      target.classList.add("vanta-on");
    } catch (err) {
      console.warn("Hero background unavailable:", err.message);
    }
  }

  // Split a heading into words and rise them in one after another.
  function words(selector, { delay = 0.1, gap = 0.06 } = {}) {
    const el = document.querySelector(selector);
    if (!el || !on()) return;
    const text = el.textContent;
    el.setAttribute("aria-label", text);
    el.innerHTML = text
      .split(" ")
      .map((w) => `<span aria-hidden="true" style="display:inline-block;white-space:pre">${w} </span>`)
      .join("");
    const spans = [...el.children];
    const anim = M().animate(
      spans,
      { opacity: [0, 1], y: [16, 0] },
      { duration: 0.7, ease: EASE, delay: M().stagger(gap, { startDelay: delay }) }
    );
    Promise.resolve(anim).then(() => {
      el.textContent = text;
      el.removeAttribute("aria-label");
    });
  }

  // Slide a marker between two positions (used by the dataset selector).
  function slide(el, from, to) {
    el.style.width = `${to.width}px`;
    if (!on() || !from || (from.left === to.left && from.width === to.width)) {
      el.style.transform = `translateX(${to.left}px)`;
      return;
    }
    el.style.transform = `translateX(${from.left}px)`;
    el.style.width = `${from.width}px`;
    const anim = M().animate(el, { x: [from.left, to.left], width: [from.width, to.width] }, { duration: 0.5, ease: EASE });
    Promise.resolve(anim).then(() => {
      el.style.transform = `translateX(${to.left}px)`;
      el.style.width = `${to.width}px`;
    });
  }

  // Cards track the cursor with a soft light (CSS reads --mx / --my).
  let spotlightOn = false;
  function spotlight() {
    if (spotlightOn || reduced()) return;
    spotlightOn = true;
    const SEL = ".panel, .kpi, .agent-card, .engine-header, .run-analysis-card";
    let raf = null;
    document.addEventListener(
      "pointermove",
      (e) => {
        if (e.pointerType !== "mouse" || raf) return;
        const card = e.target.closest && e.target.closest(SEL);
        if (!card) return;
        raf = requestAnimationFrame(() => {
          raf = null;
          const r = card.getBoundingClientRect();
          card.style.setProperty("--mx", `${e.clientX - r.left}px`);
          card.style.setProperty("--my", `${e.clientY - r.top}px`);
        });
      },
      { passive: true }
    );
  }

  return { rise, wave, count, words, slide, spotlight, background, destroyBackground, on };
})();
