  (function () {
    "use strict";

    const HOLD = 0.02;
    const TRACKS = 0.09;
    const MATCH_TOL = 1e-6;
    const LABEL_ORDER = ["entailment", "neutral", "contradiction"];
    const MAX_BYTES = 20 * 1024 * 1024;
    const SCHEMA = "jevbench.certificate.v1";
    const RUN_PATH = /^data\/[A-Za-z0-9_-]+\.json$/;

    function text(value) {
      return value == null ? "" : String(value);
    }

    function num(value) {
      const n = Number(value);
      return Number.isFinite(n) ? n : NaN;
    }

    function showLoadError(message) {
      doc = null;
      document.getElementById("verdictTitle").textContent = "Certificate refused";
      document.getElementById("verdictSub").textContent = message;
      document.getElementById("matchBanner").textContent = message;
      window.__CERTIFICATE_READY__ = true;
      window.__CERTIFICATE_MATCH__ = "✗";
    }

    function validateCertificate(data, byteLength) {
      if (byteLength > MAX_BYTES) {
        throw new Error("Certificate is larger than 20 MB.");
      }
      if (!data || typeof data !== "object" || Array.isArray(data)) {
        throw new Error("Certificate must be a JSON object.");
      }
      if (data.schema !== SCHEMA) {
        throw new Error("Unsupported schema. Expected jevbench.certificate.v1.");
      }
      for (const key of ["schema", "meta", "result", "items"]) {
        if (!(key in data)) throw new Error("Certificate is missing " + key + ".");
      }
      if (!data.meta || typeof data.meta !== "object") {
        throw new Error("Certificate meta must be an object.");
      }
      if (!data.result || typeof data.result !== "object") {
        throw new Error("Certificate result must be an object.");
      }
      if (!Array.isArray(data.items)) {
        throw new Error("Certificate items must be an array.");
      }
      return data;
    }

    let doc = null;
    let reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function probsOf(item) {
      if (!item || !Array.isArray(item.probs)) return [];
      return item.probs.map(num);
    }

    function softCorrect(item) {
      const probs = probsOf(item);
      if (!probs.length || !Array.isArray(item.human)) return NaN;
      let pick = 0;
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[pick]) pick = i;
      return num(item.human[pick]);
    }

    function topProb(item) {
      const probs = probsOf(item);
      if (!probs.length) return NaN;
      return Math.max(...probs);
    }

    /** Uniform-bin top-label ECE with optional soft correctness (mean in bin). */
    function eceSoft(items, nBins) {
      const n = items.length;
      if (!n) return NaN;
      const scores = items.map(topProb);
      const correct = items.map(softCorrect);
      const edges = [];
      for (let i = 0; i <= nBins; i++) edges.push(i / nBins);
      let ece = 0;
      for (let b = 0; b < nBins; b++) {
        const lo = edges[b], hi = edges[b + 1];
        const idx = [];
        for (let i = 0; i < n; i++) {
          const s = scores[i];
          const inBin = b === nBins - 1 ? s >= lo && s <= hi : s >= lo && s < hi;
          if (inBin) idx.push(i);
        }
        if (!idx.length) continue;
        let meanC = 0, acc = 0;
        for (const i of idx) { meanC += scores[i]; acc += correct[i]; }
        meanC /= idx.length;
        acc /= idx.length;
        ece += (idx.length / n) * Math.abs(meanC - acc);
      }
      return ece;
    }

    function recompute(doc) {
      const nBins = (doc.meta && doc.meta.n_bins) || 10;
      const easy = doc.items.filter(i => i.stratum === "easy");
      const hard = doc.items.filter(i => i.stratum === "hard");
      const ece_easy = eceSoft(easy, nBins);
      const ece_hard = eceSoft(hard, nBins);
      return { ece_easy, ece_hard, delta_ece: ece_hard - ece_easy };
    }

    function matchStatus(doc) {
      const r = recompute(doc);
      const keys = ["delta_ece", "ece_hard", "ece_easy"];
      let ok = true;
      const abs = {};
      for (const k of keys) {
        const d = Math.abs(Number(doc.result[k]) - r[k]);
        abs[k] = d;
        if (!(Number.isFinite(d) && d <= MATCH_TOL)) ok = false;
      }
      return { ok, recomputed: r, abs_delta: abs, symbol: ok ? "✓" : "✗" };
    }

    // Expose for headless tests
    window.JevbenchCertificate = {
      softCorrect, eceSoft, recompute, matchStatus, zoneFor, verdictFromHarness,
      drawGauge, getDoc: () => doc,
      // Needle tracks corrected ΔECE (same quantity as the verdict).
      gaugeNeedleDelta: () => lastGauge.corrected,
      gaugeRawDelta: () => lastGauge.raw,
    };

    let lastGauge = { corrected: NaN, raw: NaN, ciLow: NaN, ciHigh: NaN };

    function zoneFor(delta) {
      // Display gauge only. Amendment 9 verdict comes from harness result.verdict.
      if (delta <= HOLD) return "holds";
      if (delta < TRACKS) return "inconclusive";
      return "tracks";
    }

    function verdictFromHarness(result, synthetic) {
      const label = (result && result.verdict) || "inconclusive";
      if (synthetic && label === "tracks") {
        return {
          title: "Specimen: tracks accuracy (not a finding)",
          sub: "This synthetic run lands in the tracks verdict so you can see the page. It means nothing about Jev."
        };
      }
      if (label === "tracks") {
        return {
          title: "Calibration tracks accuracy",
          sub: "Harness verdict: corrected ΔECE ≥ 0.09 and parametric p < 0.05. The page does not recompute this from the interval."
        };
      }
      if (label === "holds") {
        return {
          title: "Calibration holds across difficulty",
          sub: "Harness verdict: one-sided test rejects ΔECE ≥ 0.09 and corrected ΔECE ≤ 0.02."
        };
      }
      return {
        title: "Inconclusive on the pre-registered rule",
        sub: "Harness verdict: neither tracks nor holds cleared its Amendment 9 gate."
      };
    }

    function verdictCopy(delta, synthetic) {
      // Legacy zone copy, kept for gauge labels only. Page title uses the harness.
      const z = zoneFor(delta);
      if (z === "holds") {
        return {
          title: "Calibration holds across difficulty",
          sub: "ΔECE is at or below 0.02. Hard items are not meaningfully worse-calibrated than easy ones under the pre-registered rule."
        };
      }
      if (z === "inconclusive") {
        return {
          title: "Inconclusive on the pre-registered rule",
          sub: "ΔECE sits between 0.02 and 0.09. Not a clear hold, not yet the tracks-accuracy claim."
        };
      }
      return {
        title: synthetic
          ? "Specimen: tracks accuracy (not a finding)"
          : "Calibration tracks accuracy",
        sub: synthetic
          ? "This synthetic run lands in the ≥0.09 zone so you can see the gauge. It means nothing about Jev."
          : "ΔECE is at or above 0.09. Hard items are worse-calibrated than easy ones under the pre-registered rule."
      };
    }

    function fmt(x, d) {
      if (!Number.isFinite(x)) return "n/a";
      return (x >= 0 && d >= 3 ? "+" : "") + x.toFixed(d);
    }

    function drawGauge(corrected, opts) {
      // Needle = corrected ΔECE (verdict quantity). Raw is a secondary marker.
      opts = opts || {};
      const raw = Number(opts.raw);
      const ciLow = Number(opts.ciLow);
      const ciHigh = Number(opts.ciHigh);
      const delta = Number(corrected);
      lastGauge = { corrected: delta, raw, ciLow, ciHigh };
      const svg = document.getElementById("gaugeSvg");
      const w = 420, h = 160;
      const cx = 210, cy = 140, r = 110;
      const maxX = 0.20; // gauge scale
      const clamp = (v) => Math.max(0, Math.min(maxX, Number(v) || 0));
      const ang = (v) => Math.PI + (clamp(v) / maxX) * Math.PI;
      const pt = (v, rr) => {
        const a = ang(v);
        return [cx + rr * Math.cos(a), cy + rr * Math.sin(a)];
      };
      const arc = (a0, a1, color, width) => {
        const p0 = pt(a0, r), p1 = pt(a1, r);
        const large = (ang(a1) - ang(a0)) > Math.PI ? 1 : 0;
        return `<path d="M ${p0[0]} ${p0[1]} A ${r} ${r} 0 ${large} 1 ${p1[0]} ${p1[1]}" fill="none" stroke="${color}" stroke-width="${width || 18}" />`;
      };
      const needle = pt(delta, r - 8);
      let ciBand = "";
      if (Number.isFinite(ciLow) && Number.isFinite(ciHigh) && ciHigh > ciLow) {
        const lo = clamp(ciLow), hi = clamp(ciHigh);
        ciBand = arc(lo, hi, "var(--diagonal)", 6);
      }
      let rawMark = "";
      if (Number.isFinite(raw)) {
        const rm = pt(raw, r + 2);
        const rmIn = pt(raw, r - 22);
        rawMark = `
          <line x1="${rmIn[0]}" y1="${rmIn[1]}" x2="${rm[0]}" y2="${rm[1]}" stroke="var(--bone-dim)" stroke-width="2" stroke-dasharray="3 2" />
          <circle cx="${rm[0]}" cy="${rm[1]}" r="3.5" fill="none" stroke="var(--bone-dim)" stroke-width="1.5" />
          <text x="${Math.min(400, Math.max(20, rm[0]))}" y="${Math.max(20, rm[1] - 8)}" text-anchor="middle" fill="var(--bone-dim)" font-family="var(--mono)" font-size="9">raw (biased upward)</text>
        `;
      }
      svg.innerHTML = `
        <title id="gaugeTitle">Corrected ΔECE decision gauge</title>
        <desc id="gaugeDesc">Needle at corrected ΔECE ${Number.isFinite(delta) ? delta.toFixed(3) : "n/a"}${Number.isFinite(raw) ? "; raw marker " + raw.toFixed(3) + " (biased upward)" : ""}. Zones: holds ≤0.02, inconclusive 0.02 to 0.09, tracks ≥0.09.</desc>
        ${arc(0, HOLD, "var(--hold)")}
        ${arc(HOLD, TRACKS, "var(--inconclusive)")}
        ${arc(TRACKS, maxX, "var(--tracks)")}
        ${ciBand}
        ${rawMark}
        <line x1="${cx}" y1="${cy}" x2="${needle[0]}" y2="${needle[1]}" stroke="var(--diagonal)" stroke-width="3" data-gauge-needle="corrected" />
        <circle cx="${cx}" cy="${cy}" r="6" fill="var(--diagonal)" />
        <text x="${cx}" y="28" text-anchor="middle" fill="var(--bone)" font-family="var(--mono)" font-size="22">${fmt(delta, 3)}</text>
        <text x="${cx}" y="46" text-anchor="middle" fill="var(--bone-dim)" font-size="10">corrected ΔECE</text>
        <text x="40" y="150" fill="var(--bone-dim)" font-size="11">0</text>
        <text x="200" y="48" fill="var(--bone-dim)" font-size="11">0.10</text>
        <text x="360" y="150" fill="var(--bone-dim)" font-size="11">0.20</text>
      `;
    }

    function drawScatter(items, synthetic) {
      const svg = document.getElementById("scatterSvg");
      const W = 640, H = 360, pad = 48;
      const pts = items.map(it => ({ x: topProb(it), y: softCorrect(it) }));
      // subsample for draw if huge
      let draw = pts;
      if (draw.length > 600) {
        draw = [];
        const step = Math.ceil(pts.length / 600);
        for (let i = 0; i < pts.length; i += step) draw.push(pts[i]);
      }
      const xScale = v => pad + v * (W - 2 * pad);
      const yScale = v => H - pad - v * (H - 2 * pad);
      let dots = "";
      for (const p of draw) {
        dots += `<circle cx="${xScale(p.x)}" cy="${yScale(p.y)}" r="2.2" fill="var(--jev)" opacity="0.55" />`;
      }
      svg.innerHTML = `
        <rect x="0" y="0" width="${W}" height="${H}" fill="transparent" />
        <line x1="${pad}" y1="${H-pad}" x2="${W-pad}" y2="${pad}" stroke="var(--diagonal)" stroke-dasharray="4 4" />
        <line x1="${pad}" y1="${H-pad}" x2="${W-pad}" y2="${H-pad}" stroke="var(--hairline)" />
        <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${H-pad}" stroke="var(--hairline)" />
        <text x="${W/2}" y="${H-12}" text-anchor="middle" fill="var(--bone-dim)" font-size="12">Jev top probability</text>
        <text x="16" y="${H/2}" fill="var(--bone-dim)" font-size="12" transform="rotate(-90 16 ${H/2})">Annotator share of pick</text>
        ${dots}
        ${synthetic ? `<text x="${W-12}" y="24" text-anchor="end" fill="var(--miss)" font-size="12">Synthetic</text>` : ""}
      `;
    }

    function updateGate() {
      if (!doc) return;
      const thr = Number(document.getElementById("gateThreshold").value);
      const req = Number(document.getElementById("gateAgreement").value);
      document.getElementById("gateThresholdLabel").textContent = thr.toFixed(2);
      document.getElementById("gateAgreementLabel").textContent = req.toFixed(2);
      const items = doc.items;
      const auto = items.filter(i => topProb(i) >= thr);
      const share = items.length ? auto.length / items.length : 0;
      const agree = auto.length
        ? auto.reduce((s, i) => s + softCorrect(i), 0) / auto.length
        : 0;
      // Max automate at required agreement: highest threshold? Actually:
      // among thresholds, find the lowest thr such that mean soft correct among
      // automated ≥ req, maximising share. Scan thresholds.
      let bestShare = 0;
      for (let t = 0.50; t <= 0.99; t += 0.01) {
        const sel = items.filter(i => topProb(i) >= t);
        if (!sel.length) continue;
        const a = sel.reduce((s, i) => s + softCorrect(i), 0) / sel.length;
        if (a >= req) bestShare = Math.max(bestShare, sel.length / items.length);
      }
      document.getElementById("gateShare").textContent = (100 * share).toFixed(1) + "%";
      document.getElementById("gateAgree").textContent = (100 * agree).toFixed(1) + "%";
      document.getElementById("gateMax").textContent = (100 * bestShare).toFixed(1) + "%";
    }

    function cell(className, value) {
      const td = document.createElement("td");
      if (className) td.className = className;
      td.textContent = value;
      return td;
    }

    function fillTable(items) {
      const cw = items
        .filter(i => topProb(i) >= 0.75 && softCorrect(i) < 0.35)
        .sort((a, b) => topProb(b) - topProb(a));
      const body = document.getElementById("cwBody");
      body.replaceChildren();
      const show = cw.slice(0, 85);
      for (const it of show) {
        const tr = document.createElement("tr");
        const h = Array.isArray(it.human) ? it.human.map(num) : [NaN, NaN, NaN];
        const s = h.reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0) || 1;
        tr.append(
          cell("mono", text(it.id)),
          cell("", text(it.stratum)),
          cell("mono", Number.isFinite(num(it.entropy)) ? num(it.entropy).toFixed(3) : "n/a"),
          cell("mono", topProb(it).toFixed(3)),
          cell("mono", softCorrect(it).toFixed(3)),
        );
        const td = document.createElement("td");
        const bar = document.createElement("div");
        bar.className = "human-bar";
        bar.title = "entailment / neutral / contradiction";
        bar.setAttribute("aria-label", "Annotator split");
        ["bar-e", "bar-n", "bar-c"].forEach((cls, idx) => {
          const part = document.createElement("i");
          part.className = cls;
          const share = Number.isFinite(h[idx]) ? (100 * h[idx] / s) : 0;
          part.style.width = share.toFixed(1) + "%";
          bar.appendChild(part);
        });
        td.appendChild(bar);
        tr.appendChild(td);
        body.appendChild(tr);
      }
    }

    function render(d) {
      doc = d;
      const synthetic = !!(d.meta && d.meta.synthetic);
      document.getElementById("specimenStamp").hidden = !synthetic;
      for (const id of ["gateSynthTag", "scatterSynthTag", "raceSynthTag", "tableSynthTag"]) {
        document.getElementById(id).hidden = !synthetic;
      }
      document.getElementById("modelBadge").textContent = d.meta.model_resolved || "n/a";

      const delta = Number(d.result.delta_corrected != null ? d.result.delta_corrected : d.result.delta_ece);
      const rawDelta = Number(d.result.delta_raw != null ? d.result.delta_raw : d.result.delta_ece);
      const v = verdictFromHarness(d.result, synthetic);
      document.getElementById("verdictTitle").textContent = v.title;
      document.getElementById("verdictSub").textContent = v.sub;
      drawGauge(delta, {
        raw: rawDelta,
        ciLow: d.result.ci_low,
        ciHigh: d.result.ci_high,
      });

      document.getElementById("deltaValue").textContent = fmt(delta, 3);
      if (d.result.p_value != null && Number.isFinite(Number(d.result.p_value))) {
        document.getElementById("ciValue").textContent =
          `p=${Number(d.result.p_value).toFixed(4)} · raw ${fmt(Number(d.result.delta_raw ?? d.result.delta_ece), 3)} · corr ${fmt(Number(d.result.delta_corrected ?? d.result.delta_ece), 3)}`;
      } else {
        const ciLo = Number(d.result.ci_low), ciHi = Number(d.result.ci_high);
        document.getElementById("ciValue").textContent =
          `CI [${fmt(ciLo, 3)}, ${fmt(ciHi, 3)}]`.replace(/\+\-/g, "-");
      }

      const cov = Number(d.meta.coverage_empirical);
      const covEl = document.getElementById("covValue");
      const covHint = document.getElementById("covHint");
      if (Number.isFinite(cov)) {
        covEl.textContent = (100 * cov).toFixed(1) + "%";
        if (cov < 0.95) {
          covEl.classList.add("bad");
          covHint.textContent = "below 95%";
          covHint.className = "hint miss";
        } else {
          covEl.classList.remove("bad");
          covHint.textContent = "at or above 95%";
          covHint.className = "hint ok";
        }
      }

      document.getElementById("ecePair").textContent =
        `${Number(d.result.ece_hard).toFixed(4)} / ${Number(d.result.ece_easy).toFixed(4)}`;
      document.getElementById("binningHint").textContent =
        `${d.meta.binning || "uniform"} · M=${d.meta.n_bins || 10}`;
      document.getElementById("intervalMethod").textContent = d.meta.interval_method || "n/a";
      document.getElementById("preregHint").textContent =
        `prereg ${d.meta.prereg_commit || "n/a"}`;

      const m = matchStatus(d);
      const banner = document.getElementById("matchBanner");
      banner.replaceChildren();
      const strong = document.createElement("strong");
      if (m.ok) {
        banner.classList.remove("fail");
        strong.className = "ok";
        strong.textContent = "Match check " + m.symbol;
        banner.append(strong, document.createTextNode(" page ECE from items agrees with harness result within 1e-6."));
      } else {
        banner.classList.add("fail");
        strong.className = "miss";
        strong.textContent = "Match check " + m.symbol;
        const delta = Number(m.abs_delta.delta_ece);
        const shown = Number.isFinite(delta) ? delta.toExponential(2) : "n/a";
        banner.append(strong, document.createTextNode(" page and harness disagree about ECE. Do not publish. delta=" + shown));
      }

      drawScatter(d.items, synthetic);
      updateGate();
      fillTable(d.items);
      window.__CERTIFICATE_MATCH__ = m.symbol;
      window.__CERTIFICATE_READY__ = true;
    }

    function replayTiming() {
      if (!doc || !doc.items.length) return;
      const sample = doc.items.slice(0, 40);
      const j = sample.reduce((s, i) => s + Number(i.latency_ms || 0), 0) / sample.length;
      const b = sample.reduce((s, i) => s + Number((i.baseline && i.baseline.latency_ms) || 0), 0) / sample.length;
      document.getElementById("latJev").textContent = Math.round(j) + " ms";
      document.getElementById("latBase").textContent = Math.round(b) + " ms";
      const max = Math.max(j, b, 1);
      const barJ = document.getElementById("barJev");
      const barB = document.getElementById("barBase");
      const dur = reducedMotion ? 0 : Math.min(2400, Math.max(j, b));
      barJ.style.transition = reducedMotion ? "none" : `width ${j / max * dur}ms linear`;
      barB.style.transition = reducedMotion ? "none" : `width ${b / max * dur}ms linear`;
      barJ.style.width = "0%";
      barB.style.width = "0%";
      requestAnimationFrame(() => {
        barJ.style.width = (100 * j / max) + "%";
        barB.style.width = (100 * b / max) + "%";
      });
    }

    document.getElementById("gateThreshold").addEventListener("input", updateGate);
    document.getElementById("gateAgreement").addEventListener("input", updateGate);
    document.getElementById("replayBtn").addEventListener("click", replayTiming);
    document.getElementById("fileInput").addEventListener("change", async (e) => {
      const f = e.target.files && e.target.files[0];
      if (!f) return;
      try {
        if (f.size > MAX_BYTES) throw new Error("Certificate is larger than 20 MB.");
        const raw = await f.text();
        render(validateCertificate(JSON.parse(raw), f.size));
      } catch (err) {
        showLoadError(err && err.message ? err.message : "Could not read that file.");
      }
    });
    document.getElementById("themeBtn").addEventListener("click", () => {
      const cur = document.documentElement.getAttribute("data-theme");
      const next = cur === "light" ? "dark" : cur === "dark" ? "light" : (window.matchMedia("(prefers-color-scheme: light)").matches ? "dark" : "light");
      document.documentElement.setAttribute("data-theme", next);
    });

    let resizeTimer = null;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        if (!doc) return;
        const corr = Number(doc.result.delta_corrected != null ? doc.result.delta_corrected : doc.result.delta_ece);
        const raw = Number(doc.result.delta_raw != null ? doc.result.delta_raw : doc.result.delta_ece);
        drawGauge(corr, { raw, ciLow: doc.result.ci_low, ciHigh: doc.result.ci_high });
        drawScatter(doc.items, !!(doc.meta && doc.meta.synthetic));
      }, 100);
    });

    function runPathFromQuery() {
      const params = new URLSearchParams(location.search);
      if (!params.has("run")) return "data/specimen.json";
      const raw = params.get("run") || "";
      if (RUN_PATH.test(raw)) return raw;
      return null;
    }

    async function boot() {
      const src = runPathFromQuery();
      if (!src) {
        showLoadError("The run parameter must be a same-origin path like data/name.json.");
        return;
      }
      try {
        const res = await fetch(src, { cache: "no-store" });
        if (!res.ok) throw new Error("Could not load " + src + " (" + res.status + ").");
        const raw = await res.text();
        const bytes = new TextEncoder().encode(raw).length;
        render(validateCertificate(JSON.parse(raw), bytes));
      } catch (err) {
        showLoadError(err && err.message ? err.message : "Could not load the certificate.");
      }
    }
    boot();
  })();
  