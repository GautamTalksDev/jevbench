# FAQ

## Is this affiliated with TypeSafe?

No. Independent study. Not affiliated with or endorsed by TypeSafe AI.

## Why no results yet?

The analysis was locked before the paid run. EXP-1 has not been run. Publishing a chart first and a method second would make the preregistration decorative. The files under `results/harness_fixture/` only prove the offline pipeline repeats.

## Why can't I see the sentences?

ChaosNLI is CC BY-NC. The underlying SNLI and MNLI sentences have their own licences, and they are not all compatible with this MIT tree. The repository stores ids, counts, and entropy. A fetch script can rebuild a local cache after you accept those terms.

## Can I rerun it?

Yes, offline, with `make reproduce`. That does not call Jev. A live rerun needs your own API key, the locked data hashes, and the budget cap in the experiment file ($1.00 hard cap for EXP-1). Do that after the preregistration DOI exists, and keep any deviation in the replication report.

## What does "tracks" mean?

It is one of the three pre-registered verdict labels. "Tracks" means the corrected hard-minus-easy calibration gap cleared the locked threshold and the locked statistical test. "Holds" means the gap was small enough to reject that threshold. Anything else is inconclusive. None of these labels has been earned by a live EXP-1 run.

## What if Jev passes?

Passing means the pre-registered "holds" verdict: calibration does not get worse on the hard stratum under the locked rule. That result gets published too. The preregistration already says a result that supports TypeSafe's claims will be published, and so will a null.
