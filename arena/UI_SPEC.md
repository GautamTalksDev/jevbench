# Arena UI Spec

A benchmark instrument, not a SaaS dashboard. As of the certificate page
(`index.html`), the organising artefact is a **certificate** — a plain-language
verdict from the pre-registered ΔECE rule, with empirical coverage as a headline
field and a Gate for operational decisions. Live API mode is gone; timing is a
replay of recorded latencies.

Tokens live in [`tokens.json`](tokens.json) (shared with `jevbench/charts.py`).

---

## 4.1 What this is

Lab certificate. Not a subscription dashboard. Not a live API console.

## 4.2 The organising idea: verdict, then bins

The top of the page is the verdict in plain words, then the result on a gauge
with decision zones (≤0.02 holds, 0.02–0.09 inconclusive, ≥0.09 tracks
accuracy). Calibration charts remain bin-aware: occupancy and soft-correctness
against people, not identical rounded cards.

**Avoid:** identical rounded cards in a grid, soft grey drop shadows, gradient
washes, tracked-out all-caps eyebrows, arrows on buttons, hero built from one
big number and a gradient, live API key fields.

## 4.3 Tokens

```json
{
  "color": {
    "ground":   "#2E3439",
    "panel":    "#373E44",
    "recess":   "#252A2E",
    "bone":     "#D9D4C7",
    "bone-dim": "#8E8A80",
    "jev":      "#57C4C9",
    "baseline": "#D8A24A",
    "miss":     "#C4574F",
    "diagonal": "#F2EDE1"
  }
}
```

Mid-tone slate ground (not near-black): survives video compression; avoids the
generic dark+neon generated-UI look. Light theme is supported via
`prefers-color-scheme` / Theme toggle.

## 4.4 Type

- **Archivo** (variable) — all textual UI.
- **JetBrains Mono** — numerals only (probabilities, latencies, ΔECE).
- Scale (1.25): 13 / 16 / 20 / 25 / 31. Body &lt; 80 chars/line. Sentence case.

## 4.5 Certificate rules

- Empirical coverage is a headline field; below 95% shows in red with “below 95%”.
- The page recomputes ECE from items only to confirm a match with harness
  `result.*` within 1e-6. Disagreement is loud.
- Gate is the centrepiece for humans: threshold → share automated, agreement,
  max automate at required agreement.
- “Confidence against people” places each item by Jev top probability vs
  annotator share of that pick (ChaosNLI).
- No sentence text (CC BY-NC). Items are IDs, entropy, annotator bars.
- Specimen runs stamp *Specimen* / *Synthetic*; those marks disappear when
  `"synthetic": false`. Do not record b-roll from a specimen.

## 4.6 Motion

Timing replay: bars grow over recorded latencies (including network path).
`prefers-reduced-motion` → snap.

## 4.7 Copy

Short, active, literal.

- Empty: `No run loaded. Pick a certificate.json via Load run file.`
- Buttons: `Load run file`, `Replay timing`, `Theme`
- Never: "Powered by", emoji sparkles, live API key prompts

## 4.8 Video capture

- Build at 1920×1080 CSS pixels when capturing
- Never capture a specimen gauge as if it were a finding

## Write policy

The Arena **cannot write scored `runs/<id>/`**. It reads
`data/specimen.json` or a user-loaded `certificate.json` from
`jevbench export-certificate`.
