# Glossary

Short definitions for this repository. The binding analysis is still [`PREREGISTRATION.md`](../PREREGISTRATION.md).

**Bootstrap.** Redraw the sample many times, recompute a statistic, and use the spread of those recomputations as an interval. This study's comparative claims use 10,000 resamples.

**ChaosNLI.** A dataset of natural-language inference items, each labelled by about 100 people. Disagreement among those people is the difficulty signal here.

**Delta ECE (ΔECE).** ECE on the hard stratum minus ECE on the easy stratum. Positive means hard items are worse calibrated than easy items.

**ECE (expected calibration error).** A summary of how far predicted confidence sits from the observed rate of being correct, averaged over bins of confidence.

**Entropy.** A number that grows when annotators split across labels and shrinks when they agree. Higher entropy is the hard stratum.

**Equal-n.** The same number of items in each stratum. Here that number is 750 for the primary easy and hard sets.

**EXP-1.** The pre-registered difficulty-calibration experiment. It has not been run.

**Hard scoring.** The model's top label is correct only if it matches the majority human label.

**Jev.** TypeSafe's System One model. Scored runs pin `jev-1.13.0`, not a floating alias.

**Native serving path.** Calls go to TypeSafe's API directly. Other gateways are not treated as the same measurement.

**Noul.** A Jev output that is a probability distribution. Calibration uses this (or `probabilities`), not the `confidence` sharpness score.

**Preregistration.** The hypotheses, metrics, and decision rules written down before scored calls, plus the SHA-256 hashes of the locked data files.

**Soft scoring.** The model's top label is credited with the share of annotators who chose that label, not with a single yes or no.

**Stratum.** A difficulty group. EXP-1's primary comparison uses two: easy and hard.

**Tracks.** The pre-registered name for the verdict that calibration moves with difficulty (corrected ΔECE at or above 0.09, with the parametric test the preregistration specifies). It is not a claim that this has been observed. EXP-1 has not been run.
