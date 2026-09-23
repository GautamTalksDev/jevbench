# Arena

Single-page **lab instrument** for jevbench — measures and prints, does not
sell. **No build step.** Spec: [`UI_SPEC.md`](UI_SPEC.md).

```bash
cd arena && python3 -m http.server 8765
# http://127.0.0.1:8765/
# ?chrome=off   hide nav for full-bleed capture
# ?slow=4       slow baseline crawl; on-screen "4× slowed" badge
```

Build for capture at 1920×1080 CSS pixels.

## Modes

| Mode | Needs key? | Writes scored runs? |
|---|---|---|
| **Replay** (default) | No | **Never** — reads `data/*.json` only |
| **Live** | Optional (memory only) | **Never** — badge **Demo — not scored**; download → `runs/demo/` only |

## Structural constraint

The Arena **cannot write** scored `runs/<run_id>/`. No code path targets those
folders. The harness is the source of truth.

## Views

1. **Race** (live) — Jev snaps in one frame; baseline types at token rate
2. **Calibration** (replay) — square reliability plot; occupancy **bins** under a shared x-axis (width = count)
3. **Gate** (replay) — drag threshold → % automated vs accuracy
4. **Items** (replay) — default: high confidence & wrong

## Tokens / type

Slate lab palette in [`tokens.json`](tokens.json) (shared with `charts.py`).
Archivo for text (width axis for narrow bins); JetBrains Mono for numerals only.
