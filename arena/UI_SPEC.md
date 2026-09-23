# Arena UI Spec

A benchmark instrument, not a SaaS dashboard. The reference is a laboratory
signal analyser — something that measures and prints — not an analytics product.
Every visual decision should read as "this thing is taking a measurement," not
"this thing is selling me a subscription."

The audience is skeptical senior developers who arrived from a video and are
looking for a reason not to believe you. The design's job is to make the
evidence **inspectable**, not impressive.

Tokens live in [`tokens.json`](tokens.json) (shared with `jevbench/charts.py`).

---

## 4.1 What this is

Lab instrument. Not a subscription dashboard.

## 4.2 The organising idea: bins, not cards

Calibration is about bins — sort predictions by confidence and check each
bucket against reality. The interface is built out of **bins** rather than cards.

Panels are variable-width columns whose width encodes occupancy. A bin holding
50 of 60 predictions is physically wide; a bin holding 2 is a sliver. The layout
itself carries the data, so the caveat that a crowded bin can fake a good ECE
is visible before anyone reads a number.

**Avoid:** identical rounded cards in a grid, soft grey drop shadows, gradient
washes, tracked-out all-caps eyebrows, arrows on buttons, hero built from one
big number and a gradient.

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
generic dark+neon generated-UI look. `jev` / `baseline` are cool/warm opposites
(colour-blind safer). `diagonal` is the brightest value — perfect calibration
is the truth everything is measured against.

## 4.4 Type

- **Archivo** (variable) — all textual UI. Width axis: labels condense as bins narrow.
- **JetBrains Mono** — numerals only (probabilities, latencies, costs). Tabular figures.
- Mono is **not** for labels/headings (template tell).
- Scale (1.25): 13 / 16 / 20 / 25 / 31 / 39 / 49. Body &lt; 80 chars/line. Sentence case.

## 4.5 Layout

Left-aligned throughout. Model version badge always in the header (`jev-1.13.0`),
not buried in a footer — `jev-latest` is a moving alias.

Calibration / Gate: reliability plot is a fixed square; occupancy bins share
the same x-axis underneath. Never let the axes drift.

## 4.6 Motion

**One** orchestrated moment: the Race. Jev bars arrive in a single frame (no
easing). Baseline types at real token rate (or `?slow=N`). Everything else:
motion only confirms user action.

`prefers-reduced-motion` → baseline snaps; elapsed time still shown.

## 4.7 Copy

Short, active, literal.

- Empty: `No run loaded. Pick a run from runs/.`
- Error: `Rate limited. Retrying in 4s (attempt 2 of 5).`
- Live badge: `Demo — not scored` (exactly, everywhere)
- Buttons: `Run`, `Re-run with new state`, `Copy raw JSON`
- Never: "Powered by", emoji sparkles, "blazing fast"

## 4.8 Video capture

- Build at 1920×1080 CSS pixels
- `?chrome=off` — hide nav for full-bleed b-roll
- `?slow=4` — slow baseline crawl; badge **4× slowed** on screen; disclose in VO

## Write policy

The Arena **cannot write scored `runs/<id>/`**. Replay reads `data/`. Live mode
downloads DEMO artifacts for `runs/demo/` only.
