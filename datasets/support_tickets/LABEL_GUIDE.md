# Label guide — support ticket department triage

**Task:** Assign each ticket to exactly one department label **and** exactly
one difficulty tier. Tier is a property of the **item**, assigned **before**
any model output exists. Never re-tier or re-label after seeing predictions.

| Label | Meaning |
|---|---|
| `billing` | Charges, invoices, refunds, taxes, seat billing |
| `technical` | Bugs, outages, login failures, webhooks, product defects |
| `other` | Sales questions, feedback with no ask, unclear ownership, chit-chat |

Allowed labels: `{billing, technical, other}` only.

---

## Department rubric (apply first)

1. **Primary ask wins.** If the customer asks for a refund, label `billing`
   even if they also mention a bug that caused the charge dispute.
2. **Access / crash / deploy regressions / webhooks / API errors** → `technical`.
3. **Plan / pricing / feature-availability questions without a charge dispute**
   → `other` (sales-adjacent).
4. **No actionable ask** (pure feedback, "just checking", compliments) → `other`.
5. **Multi-issue tickets:** pick the department that must act **first** to
   unblock the customer. If truly tied, prefer `billing` when money is
   disputed; otherwise `technical`. Log the item in `DISPUTED.md`.

---

## Tier criteria (concrete — a stranger must reproduce)

Tier is assigned from the **ticket text alone**, using the checklist below.
Do **not** use model confidence, entropy, or whether you personally find the
department label hard — those are contamination.

### `trivial`

All of the following hold:

- Exactly **one** clear ask (refund, password reset, crash report, etc.).
- Department is recoverable from **keywords alone** without resolving
  conflicting cues (e.g. "double charge… refund", "password reset… cannot log in").
- No second issue, no urgency framing that points at a different team, no
  dangling "also…".

Examples that belong here: single-line refund request; single-line login
failure; single-line "app crashes on export".

### `easy`

All of the following hold:

- Department is clear after **one careful read** of the full ticket.
- May need light context (which plan, which deploy) but not adjudication of
  competing asks.
- At most one secondary detail that does **not** change the department.

Examples: crash tied to a named deploy; plan-feature question that resolves to
`other` once docs vs sales conflict is noted as a sales/docs issue (not a
billing dispute).

### `hard`

At least one of the following holds:

- **Multi-issue:** two or more asks that could route to different departments
  (e.g. seat charge + stuck webhooks).
- **Urgency / framing trap:** language that nudges toward the wrong team
  (e.g. "Urgent: production is fine but CFO wants a VAT invoice" — money, not
  ops).
- Requires applying the multi-issue rule (primary-ask / first-unblock) rather
  than keyword matching.

If you used the multi-issue rule or noticed a framing trap, the tier is
`hard` (or `ambiguous` if the department itself is unclear — see below).

### `ambiguous`

At least one of the following holds:

- A second trained labeler could reasonably pick a **different department**
  under this guide (document in `DISPUTED.md`).
- No actionable ask, or the ask is so vague that department is underdetermined
  ("just checking", pure feedback with no request).
- Competing cues that the multi-issue rule does **not** resolve cleanly.

Ambiguous items still receive a forced department label (best judgment) **and**
must be listed in `DISPUTED.md` with a one-line reason.

---

## Pooling for the EXP-1 endpoint (do not label these — analysis only)

| Stratum | Tiers |
|---|---|
| **hard stratum** | `hard` ∪ `ambiguous` |
| **easy stratum** | `trivial` ∪ `easy` |

Equal **n per stratum** is enforced at analysis time
(`subsample_to_equal_n`). Keep four-tier labels for the descriptive figure.

Target from power analysis (pessimistic corner): **n = 750 per stratum**.

---

## Double-pass agreement

A random 15% subset is labelled by a **second** labeler (`labels_pass2.jsonl`).
Cohen's κ must be ≥ 0.6 before any paid run. If κ < 0.6: tighten this guide,
re-label, and say so — do not hope.

---

## What not to do

- Do not open `runs/`, model logs, Arena, or any file containing predictions
  while labelling. The labeller CLI is structurally incapable of showing
  model output — do not bypass it.
- Do not invent labels outside `{billing, technical, other}`.
- Do not invent tiers outside `{trivial, easy, hard, ambiguous}`.
- Do not use confidence, probability, or model hints when labelling.
- Do not change a label or tier after seeing model output (protocol breach).
