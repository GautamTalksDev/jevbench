# EXP-2. Decomposition (methods note)

## Honest claim

> Jev plus N labelled examples outperforms Jev alone AND outperforms a
> zero-shot frontier LLM, at a fraction of the cost.

Arm A (Jev direct) is **zero-shot**. Arm B (decomposed signals → logistic
regression) is **supervised**. They are not comparable systems. Presenting
“B beats A” alone as “Jev is better than it looks” is a misrepresentation.

## Arms

| Arm | System | Shot |
|---|---|---|
| A | Jev, one direct verdict question | zero-shot |
| B | Jev, N decomposed signals → logistic | supervised |
| C | LLM via TypeSafe adapter, direct verdict | zero-shot |
| D | LLM → logistic on the same signal schema, same N | supervised |
| E | Regex / keyword / majority floor | no model |

**A vs C** is the zero-shot comparison. **B vs D** is the supervised
comparison. Both go in the paper. Reporting only B vs C is the rigged version.

## Leakage controls

- Nested CV: outer folds evaluate; inner folds choose `C` only.
- ECE for combiners is measured on **outer held-out folds only**.
- Folds stratified by `tier|label`.
- Decomposed question text is frozen in
  `experiments/exp2_decomposition.yaml` before scored results.
- Fixed seed; fold-level variance reported alongside the mean.

## Label efficiency

Sweep N ∈ {25, 50, 100, 200} and plot held-out Arm B accuracy vs label count.
The practitioner number is the smallest N that beats zero-shot Arm C.
