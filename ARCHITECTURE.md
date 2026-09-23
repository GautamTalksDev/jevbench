# Architecture

The harness is the source of truth. The Arena only ever reads what the harness wrote.

## Tree

```
jevbench/
├── PREREGISTRATION.md          ← committed BEFORE the first API call
├── README.md
├── LICENSE                     ← MIT
├── API_REFERENCE.md            ← authoritative SDK surface
├── CONSTITUTION.md
├── ARCHITECTURE.md
├── pyproject.toml
├── Makefile                    ← make reproduce
│
├── datasets/
│   └── <task>/
│       ├── items.jsonl         ← {id, state, tier}
│       ├── labels.jsonl        ← {id, label, labeler, labeled_at}
│       ├── LABEL_GUIDE.md
│       └── DISPUTED.md
│
├── jevbench/
│   ├── clients/
│   │   ├── jev.py
│   │   ├── adapter.py
│   │   ├── prefill.py          ← Goedecke control (EXP-3)
│   │   └── trivial.py          ← regex / keyword / majority-class floor
│   ├── runner.py
│   ├── metrics.py              ← thin wrappers over netcal/MAPIE + occupancy
│   ├── stability.py            ← EXP-6: label vs confidence vs distribution
│   ├── cost.py
│   └── charts.py
│
├── experiments/
│   ├── exp1_difficulty_calibration.yaml
│   ├── exp2_decomposition.yaml
│   ├── exp3_moat_control.yaml
│   ├── exp4_invariants.yaml
│   ├── exp5_adversarial.yaml
│   ├── exp6_stability.yaml
│   └── exp7_risk_control.yaml
│
├── runs/                       ← raw JSONL, committed
│   ├── <run_id>/
│   │   ├── manifest.json
│   │   ├── raw.jsonl
│   │   └── scored.jsonl
│   └── demo/                   ← Arena live-mode. NEVER scored.
│
├── results/
├── paper/
│   ├── main.md
│   ├── RELATED_WORK.md
│   ├── figures/                ← symlink to results/
│   ├── appendix/
│   └── CITATION.cff
│
├── scripts/
│   └── lint_claims.py
│
└── arena/
    ├── index.html
    ├── app.jsx
    └── data/                   ← symlink/copy of runs/ + results/
```

## Non-negotiables

1. `PREREGISTRATION.md` and `labels.jsonl` land in an earlier commit than any run.
2. Per-run `manifest.json`: resolved model ID, SDK versions, pricing snapshot date,
   git SHA, serving path, geography, wall-clock start/end.
3. `raw.jsonl` is verbatim. Never post-processed in place.
4. `DISPUTED.md` exists from day one.
5. Arena replay mode (default) reads committed JSONL only. Live mode writes to
   `runs/demo/` and is labelled DEMO — never scored.
