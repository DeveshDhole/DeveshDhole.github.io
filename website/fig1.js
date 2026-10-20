/* Fig. 1 on the home page: a live 2D MHD simulation with a virtual spacecraft. */
/* 2D incompressible MHD, pseudo-spectral, periodic box [0, 2π)².
   State: flux function ψ (B = ∇ψ × ẑ) and vorticity ω (u = ∇φ × ẑ, ω = −∇²φ).
     ∂ψ/∂t = −u·∇ψ + η∇²ψ
     ∂ω/∂t = −u·∇ω + B·∇J + ν∇²ω,   J = −∇²ψ
   2/3-rule dealiasing, integrating-factor Adams–Bashforth 2 time stepping. */
function createFFT(n) {
  const levels = Math.round(Math.log2(n));
  const rev = new Uint16Array(n);
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let b = 0; b < levels; b++) r |= ((i >> b) & 1) << (levels - 1 - b);
    rev[i] = r;
  }
  const cs = new Float64Array(n / 2), sn = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) {
    cs[i] = Math.cos(2 * Math.PI * i / n);
    sn[i] = Math.sin(2 * Math.PI * i / n);
  }
  return function fft(re, im, inverse) {
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    const sgn = inverse ? 1 : -1;
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let start = 0; start < n; start += size) {
        for (let k = 0; k < half; k++) {
          const wr = cs[k * step], wi = sgn * sn[k * step];
          const a = start + k, b = a + half;
          const xr = re[b] * wr - im[b] * wi;
          const xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi;
          re[a] += xr; im[a] += xi;
        }
      }
    }
  };
}

function createMHD(N, opts) {
  const o = Object.assign({ nu: 1.5e-3, eta: 1.5e-3, nu4: 0, eta4: 0, dt: 0.005, k0: 2.5, u0: 0.6, b0: 0.6 }, opts || {});
  const NN = N * N, kc = Math.floor(N / 3), L = 2 * Math.PI, dx = L / N;
  const fft = createFFT(N);
  const tr = new Float64Array(N), ti = new Float64Array(N);
  const F = () => new Float64Array(NN);

  const KX = F(), KY = F(), K2 = F(), IK2 = F(), MASK = F();
  const NEG = new Int32Array(NN);
  const keptRows = [];
  for (let j = 0; j < N; j++) {
    const ky = j < N / 2 ? j : j - N;
    if (Math.abs(ky) <= kc) keptRows.push(j);
    for (let i = 0; i < N; i++) {
      const kx = i < N / 2 ? i : i - N, m = j * N + i;
      KX[m] = kx; KY[m] = ky; K2[m] = kx * kx + ky * ky;
      IK2[m] = K2[m] > 0 ? 1 / K2[m] : 0;
      MASK[m] = Math.abs(kx) <= kc && Math.abs(ky) <= kc && K2[m] > 0 ? 1 : 0;
      NEG[m] = ((N - j) % N) * N + ((N - i) % N);
    }
  }

  // 2D transforms. Rows with |ky| > kc are zero in spectral space, so they are skipped.
  function rowPass(re, im, inverse) {
    for (let r = 0; r < keptRows.length; r++) {
      const off = keptRows[r] * N;
      for (let i = 0; i < N; i++) { tr[i] = re[off + i]; ti[i] = im[off + i]; }
      fft(tr, ti, inverse);
      for (let i = 0; i < N; i++) { re[off + i] = tr[i]; im[off + i] = ti[i]; }
    }
  }
  function colPass(re, im, inverse) {
    for (let i = 0; i < N; i++) {
      for (let j = 0, m = i; j < N; j++, m += N) { tr[j] = re[m]; ti[j] = im[m]; }
      fft(tr, ti, inverse);
      for (let j = 0, m = i; j < N; j++, m += N) { re[m] = tr[j]; im[m] = ti[j]; }
    }
  }
  function toReal(re, im) {           // spectral -> real space (normalised)
    rowPass(re, im, true);
    colPass(re, im, true);
    const s = 1 / NN;
    for (let m = 0; m < NN; m++) { re[m] *= s; im[m] *= s; }
  }
  function toSpectral(re, im) {       // real -> spectral (rows beyond kc left stale; always masked)
    colPass(re, im, false);
    rowPass(re, im, false);
  }

  // state and work arrays
  const pr = F(), pi = F(), wr = F(), wi = F();
  let nA = [F(), F(), F(), F()], nB = [F(), F(), F(), F()];
  const aR = F(), aI = F(), bR = F(), bI = F(), cR = F(), cI = F(), dR = F(), dI = F(), eR = F(), eI = F(), gR = F(), gI = F();
  const EP = F(), EW = F();
  const sim = { N, dx, t: 0, steps: 0, seed: 0, opts: o,
    psix: aR, psiy: aI, ux: bR, uy: bI, J: gR, psi: gI, maxSpeed: 0 };

  function setFactors() {
    for (let m = 0; m < NN; m++) {
      const k2 = K2[m];
      EP[m] = Math.exp(-(o.eta * k2 + o.eta4 * k2 * k2) * o.dt);
      EW[m] = Math.exp(-(o.nu * k2 + o.nu4 * k2 * k2) * o.dt);
    }
  }

  // Nonlinear terms for the current state -> out = [Nψ re, Nψ im, Nω re, Nω im].
  // Leaves real-space ψx, ψy, ux, uy, J, ψ in the work arrays for drawing and sampling.
  function nonlinear(out) {
    for (let m = 0; m < NN; m++) {
      const kx = KX[m], ky = KY[m];
      const p_r = pr[m], p_i = pi[m], o_r = wr[m], o_i = wi[m];
      const f_r = o_r * IK2[m], f_i = o_i * IK2[m];
      const j_r = p_r * K2[m], j_i = p_i * K2[m];
      aR[m] = -kx * p_i - ky * p_r; aI[m] = kx * p_r - ky * p_i;   // ψx + iψy
      bR[m] = kx * f_r - ky * f_i;  bI[m] = ky * f_r + kx * f_i;   // ux + iuy
      cR[m] = -kx * o_i - ky * o_r; cI[m] = kx * o_r - ky * o_i;   // ωx + iωy
      dR[m] = -kx * j_i - ky * j_r; dI[m] = kx * j_r - ky * j_i;   // Jx + iJy
      gR[m] = j_r - p_i;            gI[m] = j_i + p_r;             // J + iψ
    }
    toReal(aR, aI); toReal(bR, bI); toReal(cR, cI); toReal(dR, dI); toReal(gR, gI);
    let vmax = 0;
    for (let m = 0; m < NN; m++) {
      const psx = aR[m], psy = aI[m], ux = bR[m], uy = bI[m];
      eR[m] = -(ux * psx + uy * psy);
      eI[m] = -(ux * cR[m] + uy * cI[m]) + (psy * dR[m] - psx * dI[m]);
      const v = Math.abs(ux) + Math.abs(uy) + Math.abs(psx) + Math.abs(psy);
      if (v > vmax) vmax = v;
    }
    sim.maxSpeed = vmax;
    toSpectral(eR, eI);
    const [Pr, Pi, Wr, Wi] = out;
    for (let m = 0; m < NN; m++) {
      if (MASK[m] === 0) { Pr[m] = Pi[m] = Wr[m] = Wi[m] = 0; continue; }
      const n = NEG[m], a = eR[m], b = eI[m], c = eR[n], d = eI[n];
      Pr[m] = 0.5 * (a + c); Pi[m] = 0.5 * (b - d);
      Wr[m] = 0.5 * (b + d); Wi[m] = 0.5 * (c - a);
    }
  }

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Random large-scale field with Gaussian spectral envelope, projected onto a real field.
  function randomField(rng, outR, outI) {
    const gauss = () => {
      let u = 0, v = 0;
      while (u === 0) u = rng();
      v = rng();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    for (let m = 0; m < NN; m++) {
      const k2 = K2[m];
      if (MASK[m] && k2 <= 36 * o.k0 * o.k0) {
        const amp = Math.exp(-k2 / (2 * o.k0 * o.k0));
        outR[m] = amp * gauss(); outI[m] = amp * gauss();
      } else { outR[m] = 0; outI[m] = 0; }
    }
    toReal(outR, outI);
    outI.fill(0);
    toSpectral(outR, outI);
    for (let m = 0; m < NN; m++) { outR[m] *= MASK[m]; outI[m] *= MASK[m]; }
  }

  function meanSquare(re, im, weight) {   // ⟨|∇f|²⟩-type sums via Parseval
    let s = 0;
    for (let m = 0; m < NN; m++) s += weight[m] * (re[m] * re[m] + im[m] * im[m]);
    return s / (NN * NN);
  }

  sim.energies = function () {
    const eb = meanSquare(pr, pi, K2);
    const eu = meanSquare(wr, wi, IK2);
    return { eb: 0.5 * eb, eu: 0.5 * eu, brms: Math.sqrt(eb), urms: Math.sqrt(eu) };
  };

  sim.reset = function (seed) {
    sim.seed = seed;
    const rng = mulberry32(seed * 2654435761 + 12345);
    randomField(rng, pr, pi);
    randomField(rng, wr, wi);   // this is φ̂ for now
    let sb = 0, su = 0;
    for (let m = 0; m < NN; m++) {
      sb += K2[m] * (pr[m] * pr[m] + pi[m] * pi[m]);
      su += K2[m] * (wr[m] * wr[m] + wi[m] * wi[m]);
    }
    const fb = o.b0 / Math.sqrt(sb / (NN * NN)), fu = o.u0 / Math.sqrt(su / (NN * NN));
    for (let m = 0; m < NN; m++) {
      pr[m] *= fb; pi[m] *= fb;
      wr[m] *= fu * K2[m]; wi[m] *= fu * K2[m];   // ω̂ = k² φ̂
    }
    sim.t = 0; sim.steps = 0;
    setFactors();
    nonlinear(nA);
  };

  // One IF-AB2 step. On entry nA holds N(state at t); work arrays hold real fields at t.
  sim.step = function () {
    const dt = o.dt, first = sim.steps === 0;
    const [Pr, Pi, Wr, Wi] = nA, [Pr0, Pi0, Wr0, Wi0] = nB;
    for (let m = 0; m < NN; m++) {
      const ep = EP[m], ew = EW[m];
      if (first) {
        pr[m] = ep * (pr[m] + dt * Pr[m]); pi[m] = ep * (pi[m] + dt * Pi[m]);
        wr[m] = ew * (wr[m] + dt * Wr[m]); wi[m] = ew * (wi[m] + dt * Wi[m]);
      } else {
        pr[m] = ep * (pr[m] + dt * (1.5 * Pr[m] - 0.5 * ep * Pr0[m]));
        pi[m] = ep * (pi[m] + dt * (1.5 * Pi[m] - 0.5 * ep * Pi0[m]));
        wr[m] = ew * (wr[m] + dt * (1.5 * Wr[m] - 0.5 * ew * Wr0[m]));
        wi[m] = ew * (wi[m] + dt * (1.5 * Wi[m] - 0.5 * ew * Wi0[m]));
      }
    }
    const tmp = nB; nB = nA; nA = tmp;
    sim.t += dt; sim.steps++;
    nonlinear(nA);          // refresh N and real-space fields at the new time
  };

  // Bilinear sample of a real-space field at physical position (x, y), periodic.
  sim.sample = function (field, x, y) {
    let gx = x / dx, gy = y / dx;
    gx -= Math.floor(gx / N) * N; gy -= Math.floor(gy / N) * N;
    const i0 = Math.floor(gx), j0 = Math.floor(gy), fx = gx - i0, fy = gy - j0;
    const i1 = (i0 + 1) % N, j1 = (j0 + 1) % N;
    return (1 - fy) * ((1 - fx) * field[j0 * N + i0] + fx * field[j0 * N + i1]) +
           fy * ((1 - fx) * field[j1 * N + i0] + fx * field[j1 * N + i1]);
  };

  return sim;
}

(function () {
  'use strict';
  const canvas = document.getElementById('fig1-canvas');
  if (!canvas || !canvas.getContext || typeof createMHD !== 'function') return;
  const ctx = canvas.getContext('2d');
  const btnPlay = document.getElementById('fig1-play');
  const btnNew = document.getElementById('fig1-new');
  const btnLines = document.getElementById('fig1-lines');

  // ---- settings -------------------------------------------------------------
  const N = 128;
  const SIM = { dt: 0.006, nu: 1e-3, eta: 1e-3, nu4: 4e-6, eta4: 4e-6, k0: 2.5, u0: 0.6, b0: 0.6 };
  const FIRST_SEED = 7;
  const RATE = 0.15;         // simulation time per second of wall-clock time
  const FAST_RATE = 0.5;     // quicker start, so current sheets appear within a couple of seconds
  const FAST_UNTIL = 0.6;
  const T_END = 16;          // after this, start a fresh run
  const WARM = 1.0;          // when paused, show a run from this time rather than t = 0
  const CLIP = 3.5;          // colour scale saturates at this many rms
  const GAMMA = 1.4;         // keeps weak currents pale so the sheets stand out
  const WINDOW = 4;          // time span shown in the spacecraft panels
  const TRAIL = 1.0;         // spacecraft trail length (time units)
  const B_RANGE = 2.5, J_RANGE = 6;
  const V_SC = 1.4, ANGLE = Math.atan2(0.3, 1), X_SC = 0.4, Y_SC = 1.3;
  const CAP = 2048;
  const TWO_PI = 2 * Math.PI;

  const sim = createMHD(N, SIM);
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const darkMode = window.matchMedia('(prefers-color-scheme: dark)');

  // ---- spacecraft record (ring buffer) ---------------------------------------
  const rec = {
    t: new Float64Array(CAP), x: new Float64Array(CAP), y: new Float64Array(CAP),
    bx: new Float32Array(CAP), by: new Float32Array(CAP), jz: new Float32Array(CAP),
    head: 0, size: 0
  };
  let scX = X_SC, scY = Y_SC, jrms = 1, levelStep = 1, seed = FIRST_SEED, startSpeed = 1;

  function rmsOf(a) {
    let s = 0;
    for (let m = 0; m < a.length; m++) s += a[m] * a[m];
    return Math.sqrt(s / a.length);
  }
  function sampleSpacecraft() {
    const t = sim.t;
    scX = (X_SC + V_SC * Math.cos(ANGLE) * t) % TWO_PI;
    scY = (Y_SC + V_SC * Math.sin(ANGLE) * t) % TWO_PI;
    jrms += (rmsOf(sim.J) - jrms) * 0.08;
    const brms = sim.energies().brms || 1;
    const i = rec.head;
    rec.t[i] = t; rec.x[i] = scX; rec.y[i] = scY;
    rec.bx[i] = sim.sample(sim.psiy, scX, scY) / brms;   // Bx =  ∂ψ/∂y
    rec.by[i] = -sim.sample(sim.psix, scX, scY) / brms;  // By = −∂ψ/∂x
    rec.jz[i] = sim.sample(sim.J, scX, scY) / jrms;
    rec.head = (i + 1) % CAP;
    rec.size = Math.min(rec.size + 1, CAP);
  }
  function reset(s) {
    seed = s;
    sim.reset(seed);
    jrms = rmsOf(sim.J) || 1;
    let lo = Infinity, hi = -Infinity;
    const psi = sim.psi;
    for (let m = 0; m < psi.length; m++) { if (psi[m] < lo) lo = psi[m]; if (psi[m] > hi) hi = psi[m]; }
    levelStep = (hi - lo) / 16 || 1;
    startSpeed = sim.maxSpeed || 1;
    rec.head = 0; rec.size = 0; acc = 0;
    sampleSpacecraft();
  }

  // ---- theme ------------------------------------------------------------------
  let theme = null;
  const LUT_SIZE = 512;
  const LUT = new Uint8ClampedArray(LUT_SIZE * 3);
  function hexToRgb(h) {
    h = h.replace('#', '').trim();
    if (h.length === 3) h = h.split('').map(c => c + c).join('');
    const n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function readTheme() {
    const cs = getComputedStyle(document.documentElement);
    const v = name => cs.getPropertyValue(name).trim();
    const stopsCss = ['--cm-n2', '--cm-n1', '--cm-0', '--cm-p1', '--cm-p2'].map(v);
    theme = {
      ink: v('--ink'), slate: v('--slate'), rule: v('--rule'), paper: v('--paper'),
      blue: v('--blue'), green: v('--green'), red: v('--red'), line: v('--fieldline'),
      serif: v('--serif') || 'serif', stopsCss, stops: stopsCss.map(hexToRgb)
    };
    // colour lookup over the clipped, normalised current: v in [-1, 1] -> gamma -> diverging map
    for (let b = 0; b < LUT_SIZE; b++) {
      const val = (b / (LUT_SIZE - 1)) * 2 - 1;
      const p = Math.sign(val) * Math.pow(Math.abs(val), GAMMA);
      const x = (p + 1) * 2, k = Math.min(3, Math.floor(x)), f = x - k;
      const a = theme.stops[k], c = theme.stops[k + 1];
      for (let ch = 0; ch < 3; ch++) LUT[b * 3 + ch] = a[ch] + (c[ch] - a[ch]) * f;
    }
  }

  // ---- geometry -------------------------------------------------------------
  let W = 0, H = 0, dpr = 1, G = null;
  const heat = document.createElement('canvas');
  heat.width = N; heat.height = N;
  const heatCtx = heat.getContext('2d');
  const heatImg = heatCtx.createImageData(N, N);

  function layout() {
    const w = Math.round(canvas.parentElement.getBoundingClientRect().width);
    if (!w) return false;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const padL = 46, padR = 46, padT = 26;
    const P = Math.max(150, w - padL - padR);
    const hP = Math.round(Math.max(54, Math.min(84, P * 0.18)));
    const yMap = padT, yB = yMap + P + 52, yJ = yB + hP, yEnd = yJ + hP;
    W = w; H = yEnd + 34;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    canvas.style.height = H + 'px';
    G = { x0: padL, P, yMap, yB, yJ, hP, yEnd };
    return true;
  }

  // ---- text helpers -----------------------------------------------------------
  function setFont(size, italic) {
    ctx.font = (italic ? 'italic ' : '') + '400 ' + size + 'px ' + theme.serif;
  }
  // parts: [{ t: 'J', i: true }, { t: 'z', i: true, sub: true }]
  function mathText(parts, x, y, size, color, align) {
    const sizes = parts.map(p => (p.sub ? Math.round(size * 0.72) : size));
    const widths = parts.map((p, k) => { setFont(sizes[k], p.i); return ctx.measureText(p.t).width; });
    const total = widths.reduce((a, b) => a + b, 0);
    let cx = align === 'center' ? x - total / 2 : align === 'right' ? x - total : x;
    ctx.fillStyle = color;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    parts.forEach((p, k) => {
      setFont(sizes[k], p.i);
      ctx.fillText(p.t, cx, y + (p.sub ? size * 0.3 : 0));
      cx += widths[k] + (p.i && !p.sub ? size * 0.05 : 0);
    });
  }
  function label(text, x, y, size, color, align) {
    setFont(size, false);
    ctx.fillStyle = color;
    ctx.textAlign = align;
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(text, x, y);
  }

  // ---- drawing ----------------------------------------------------------------
  function drawHeatmap() {
    const { x0, P, yMap } = G;
    const J = sim.J, d = heatImg.data, k = 1 / (CLIP * jrms), top = LUT_SIZE - 1;
    for (let j = 0; j < N; j++) {
      const row = (N - 1 - j) * N;
      for (let i = 0; i < N; i++) {
        let b = ((J[j * N + i] * k + 1) * 0.5 * top + 0.5) | 0;
        if (b < 0) b = 0; else if (b > top) b = top;
        const o = (row + i) * 4, l = b * 3;
        d[o] = LUT[l]; d[o + 1] = LUT[l + 1]; d[o + 2] = LUT[l + 2]; d[o + 3] = 255;
      }
    }
    heatCtx.putImageData(heatImg, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(heat, x0, yMap, P, P);
  }

  // Marching squares on the periodic grid: field lines are contours of ψ.
  const SEGMENTS = [null, [3, 0], [0, 1], [3, 1], [1, 2], null, [0, 2], [3, 2], [2, 3], [0, 2], null, [1, 2], [3, 1], [0, 1], [3, 0], null];
  function drawFieldLines() {
    const { x0, P, yMap } = G;
    const psi = sim.psi, s = P / N, d = levelStep, off = 0.5 * levelStep;
    let ex = 0, ey = 0;
    let i = 0, j = 0, v0 = 0, v1 = 0, v2 = 0, v3 = 0, lv = 0, wx = false, wy = false;
    function edge(e) {
      switch (e) {
        case 0: ex = i + (lv - v0) / (v1 - v0); ey = j; break;
        case 1: ex = i + 1; ey = j + (lv - v1) / (v2 - v1); break;
        case 2: ex = i + (lv - v3) / (v2 - v3); ey = j + 1; break;
        default: ex = i; ey = j + (lv - v0) / (v3 - v0);
      }
    }
    const X = gx => x0 + (gx + 0.5) * s, Y = gy => yMap + P - (gy + 0.5) * s;
    function seg(ea, eb) {
      edge(ea); const ax = ex, ay = ey;
      edge(eb); const bx = ex, by = ey;
      ctx.moveTo(X(ax), Y(ay)); ctx.lineTo(X(bx), Y(by));
      if (wx) { ctx.moveTo(X(ax - N), Y(ay)); ctx.lineTo(X(bx - N), Y(by)); }
      if (wy) { ctx.moveTo(X(ax), Y(ay - N)); ctx.lineTo(X(bx), Y(by - N)); }
      if (wx && wy) { ctx.moveTo(X(ax - N), Y(ay - N)); ctx.lineTo(X(bx - N), Y(by - N)); }
    }
    ctx.save();
    ctx.beginPath(); ctx.rect(x0, yMap, P, P); ctx.clip();
    ctx.beginPath();
    for (j = 0; j < N; j++) {
      const j1 = j + 1 === N ? 0 : j + 1;
      wy = j === N - 1;
      for (i = 0; i < N; i++) {
        const i1 = i + 1 === N ? 0 : i + 1;
        wx = i === N - 1;
        v0 = psi[j * N + i]; v1 = psi[j * N + i1]; v2 = psi[j1 * N + i1]; v3 = psi[j1 * N + i];
        const lo = Math.min(v0, v1, v2, v3), hi = Math.max(v0, v1, v2, v3);
        const n0 = Math.ceil((lo - off) / d), n1 = Math.floor((hi - off) / d);
        for (let n = n0; n <= n1; n++) {
          lv = off + n * d;
          const c = (v0 > lv ? 1 : 0) | (v1 > lv ? 2 : 0) | (v2 > lv ? 4 : 0) | (v3 > lv ? 8 : 0);
          if (c === 0 || c === 15) continue;
          if (c === 5 || c === 10) {
            const centreAbove = (v0 + v1 + v2 + v3) * 0.25 > lv;
            if ((c === 5) === centreAbove) { seg(0, 1); seg(2, 3); } else { seg(3, 0); seg(1, 2); }
          } else {
            seg(SEGMENTS[c][0], SEGMENTS[c][1]);
          }
        }
      }
    }
    ctx.strokeStyle = theme.line;
    ctx.lineWidth = 0.8;
    ctx.stroke();
    ctx.restore();
  }

  function drawSpacecraft() {
    const { x0, P, yMap } = G;
    const s = P / N, dx = sim.dx;
    const cx = x => x0 + ((x / dx + 0.5) % N) * s;
    const cy = y => yMap + P - ((y / dx + 0.5) % N) * s;
    ctx.save();
    ctx.beginPath(); ctx.rect(x0, yMap, P, P); ctx.clip();
    ctx.beginPath();
    let px = NaN, py = NaN;
    for (let n = rec.size - 1; n >= 0; n--) {
      const i = (rec.head - 1 - n + CAP) % CAP;
      if (rec.t[i] < sim.t - TRAIL) continue;
      const x = cx(rec.x[i]), y = cy(rec.y[i]);
      if (!(Math.abs(x - px) < P / 2 && Math.abs(y - py) < P / 2)) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      px = x; py = y;
    }
    ctx.globalAlpha = 0.7;
    ctx.strokeStyle = theme.ink;
    ctx.lineWidth = 1.4;
    ctx.setLineDash([2, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    const mx = cx(scX), my = cy(scY);
    ctx.beginPath();
    ctx.arc(mx, my, 4.5, 0, TWO_PI);
    ctx.lineWidth = 3;
    ctx.strokeStyle = theme.paper;
    ctx.stroke();
    ctx.fillStyle = theme.ink;
    ctx.fill();
    ctx.restore();
  }

  function drawMapAxes() {
    const { x0, P, yMap } = G;
    ctx.strokeStyle = theme.ink;
    ctx.lineWidth = 1;
    ctx.strokeRect(x0 + 0.5, yMap + 0.5, P - 1, P - 1);
    ctx.beginPath();
    for (const f of [0.25, 0.5, 0.75]) {
      const len = f === 0.5 ? 6 : 4;
      const x = Math.round(x0 + f * P) + 0.5, y = Math.round(yMap + P - f * P) + 0.5;
      ctx.moveTo(x, yMap + P); ctx.lineTo(x, yMap + P - len);
      ctx.moveTo(x, yMap); ctx.lineTo(x, yMap + len);
      ctx.moveTo(x0, y); ctx.lineTo(x0 + len, y);
      ctx.moveTo(x0 + P, y); ctx.lineTo(x0 + P - len, y);
    }
    ctx.stroke();
    const ticks = [[0, '0'], [0.5, 'π'], [1, '2π']];
    for (const [f, s] of ticks) label(s, x0 + f * P, yMap + P + 18, 13, theme.slate, 'center');
    for (const [f, s] of ticks) label(s, x0 - 8, yMap + P - f * P + 4.5, 13, theme.slate, 'right');
    mathText([{ t: 'x', i: true }], x0 + P / 2, yMap + P + 38, 15, theme.ink, 'center');
    mathText([{ t: 'y', i: true }], x0 - 37, yMap + P / 2 + 5, 15, theme.ink, 'center');
    mathText([{ t: 't', i: true }, { t: ' = ' + sim.t.toFixed(2) }], x0, yMap - 10, 14, theme.slate, 'left');
  }

  function drawColorbar() {
    const { x0, P, yMap } = G;
    const bx = x0 + P + 12, bw = 9;
    const g = ctx.createLinearGradient(0, yMap + P, 0, yMap);
    // the gradient follows the same gamma curve as the map
    for (let q = 0; q <= 32; q++) {
      const b = Math.round((q / 32) * (LUT_SIZE - 1)) * 3;
      g.addColorStop(q / 32, 'rgb(' + LUT[b] + ',' + LUT[b + 1] + ',' + LUT[b + 2] + ')');
    }
    ctx.fillStyle = g;
    ctx.fillRect(bx, yMap, bw, P);
    ctx.strokeStyle = theme.ink;
    ctx.lineWidth = 1;
    ctx.strokeRect(bx + 0.5, yMap + 0.5, bw - 1, P - 1);
    ctx.beginPath();
    for (const v of [-3, 0, 3]) {
      const y = Math.round(yMap + P / 2 - (v / CLIP) * (P / 2)) + 0.5;
      ctx.moveTo(bx + bw, y); ctx.lineTo(bx + bw + 4, y);
      label(v < 0 ? '−' + -v : String(v), bx + bw + 7, y + 4.5, 13, theme.slate, 'left');
    }
    ctx.stroke();
    mathText([{ t: 'J', i: true }, { t: 'z', i: true, sub: true }], bx + bw / 2 + 2, yMap - 10, 15, theme.ink, 'center');
  }

  function drawPanels() {
    const { x0, P, yB, yJ, hP, yEnd } = G;
    const tNow = sim.t, t0 = Math.max(0, tNow - WINDOW);
    const X = t => x0 + ((t - t0) / WINDOW) * P;

    ctx.strokeStyle = theme.rule;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const yc of [yB + hP / 2, yJ + hP / 2]) { ctx.moveTo(x0, Math.round(yc) + 0.5); ctx.lineTo(x0 + P, Math.round(yc) + 0.5); }
    ctx.stroke();

    function trace(arr, y0, range, color) {
      ctx.save();
      ctx.beginPath(); ctx.rect(x0 + 1, y0 + 1, P - 2, hP - 2); ctx.clip();
      ctx.beginPath();
      let started = false;
      for (let n = rec.size - 1; n >= 0; n--) {
        const i = (rec.head - 1 - n + CAP) % CAP;
        if (rec.t[i] < t0) continue;
        const x = X(rec.t[i]), y = y0 + hP / 2 - (arr[i] / range) * (hP / 2);
        if (started) ctx.lineTo(x, y); else { ctx.moveTo(x, y); started = true; }
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.3;
      ctx.lineJoin = 'round';
      ctx.stroke();
      ctx.restore();
    }
    trace(rec.bx, yB, B_RANGE, theme.blue);
    trace(rec.by, yB, B_RANGE, theme.green);
    trace(rec.jz, yJ, J_RANGE, theme.red);

    ctx.strokeStyle = theme.ink;
    ctx.lineWidth = 1;
    ctx.strokeRect(x0 + 0.5, yB + 0.5, P - 1, 2 * hP - 1);
    ctx.beginPath();
    ctx.moveTo(x0, yJ + 0.5); ctx.lineTo(x0 + P, yJ + 0.5);
    // y ticks
    const yTicks = [[yB, B_RANGE, 1], [yJ, J_RANGE, 3]];
    for (const [y0, range, v] of yTicks) {
      for (const val of [-v, 0, v]) {
        const y = Math.round(y0 + hP / 2 - (val / range) * (hP / 2)) + 0.5;
        ctx.moveTo(x0, y); ctx.lineTo(x0 + 4, y);
        label(val < 0 ? '−' + -val : String(val), x0 - 7, y + 4.5, 12, theme.slate, 'right');
      }
    }
    // time ticks
    for (let k = Math.ceil(t0); k <= Math.floor(t0 + WINDOW + 1e-9); k++) {
      const x = Math.round(X(k)) + 0.5;
      ctx.moveTo(x, yEnd); ctx.lineTo(x, yEnd - 5);
      ctx.moveTo(x, yB); ctx.lineTo(x, yB + 4);
      ctx.moveTo(x, yJ - 3); ctx.lineTo(x, yJ + 3);
      label(String(k), x, yEnd + 18, 13, theme.slate, 'center');
    }
    ctx.stroke();

    mathText([{ t: 'B', i: true }, { t: 'x', i: true, sub: true }], x0 + P + 10, yB + hP / 2 - 5, 15, theme.blue, 'left');
    mathText([{ t: 'B', i: true }, { t: 'y', i: true, sub: true }], x0 + P + 10, yB + hP / 2 + 14, 15, theme.green, 'left');
    mathText([{ t: 'J', i: true }, { t: 'z', i: true, sub: true }], x0 + P + 10, yJ + hP / 2 + 5, 15, theme.red, 'left');
    mathText([{ t: 't', i: true }], x0 + P + 14, yEnd + 18, 15, theme.ink, 'left');
  }

  function draw() {
    if (!G || !theme) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    drawHeatmap();
    if (showLines) drawFieldLines();
    drawSpacecraft();
    drawMapAxes();
    drawColorbar();
    drawPanels();
  }

  // ---- running ----------------------------------------------------------------
  let running = false, loopId = 0, userPaused = false, onScreen = true, lastNow = 0, acc = 0, showLines = true, fading = false;

  function shouldRun() { return !userPaused && onScreen && !document.hidden; }
  function update() {
    if (shouldRun()) {
      if (!running) {
        running = true;
        lastNow = performance.now();
        const id = ++loopId;
        requestAnimationFrame(now => frame(now, id));
      }
    } else {
      running = false;
    }
  }
  function frame(now, id) {
    if (!running || id !== loopId) return;
    const wall = Math.min(0.1, Math.max(0, (now - lastNow) / 1000));
    lastNow = now;
    acc += wall * (sim.t < FAST_UNTIL ? FAST_RATE : RATE);
    const start = performance.now();
    let steps = 0;
    while (acc >= SIM.dt) {
      sim.step();
      sampleSpacecraft();
      acc -= SIM.dt;
      steps++;
      if (performance.now() - start > 14) { acc = 0; break; }  // slow device: run slower rather than stutter
    }
    if (!isFinite(sim.maxSpeed) || sim.maxSpeed > 4 * startSpeed) newRun(false);
    else if (sim.t > T_END) newRun(true);
    if (steps) draw();
    requestAnimationFrame(t => frame(t, id));
  }
  function warmUp(target) {
    if (running || sim.t >= target) { draw(); return; }
    const start = performance.now();
    while (sim.t < target && performance.now() - start < 12) { sim.step(); sampleSpacecraft(); }
    setTimeout(() => warmUp(target), 0);
  }
  function newRun(animate) {
    if (fading) return;
    const go = () => {
      reset(seed + 1);
      draw();
      if (!running) warmUp(WARM);
    };
    if (!animate || reduceMotion.matches || !running) { go(); return; }
    fading = true;
    canvas.style.opacity = '0';
    setTimeout(() => { go(); canvas.style.opacity = '1'; fading = false; }, 350);
  }

  // ---- controls and environment ---------------------------------------------------
  btnPlay.addEventListener('click', () => {
    userPaused = !userPaused;
    btnPlay.textContent = userPaused ? 'Play' : 'Pause';
    update();
  });
  btnNew.addEventListener('click', () => newRun(true));
  btnLines.addEventListener('click', () => {
    showLines = !showLines;
    btnLines.textContent = showLines ? 'Hide field lines' : 'Show field lines';
    draw();
  });
  document.addEventListener('visibilitychange', update);
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(entries => {
      onScreen = entries[entries.length - 1].isIntersecting;
      update();
    }, { threshold: 0.02 }).observe(canvas);
  }
  if ('ResizeObserver' in window) {
    new ResizeObserver(() => { if (layout()) draw(); }).observe(canvas.parentElement);
  } else {
    window.addEventListener('resize', () => { if (layout()) draw(); });
  }
  const onTheme = () => { readTheme(); draw(); };
  if (darkMode.addEventListener) darkMode.addEventListener('change', onTheme); else darkMode.addListener(onTheme);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(draw);

  readTheme();
  layout();
  reset(FIRST_SEED);
  draw();
  if (reduceMotion.matches) {
    userPaused = true;
    btnPlay.textContent = 'Play';
    warmUp(WARM);
  }
  update();
})();
