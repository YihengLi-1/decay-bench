# Pre-registered analysis plan

Written 2026-10-25, before the confirmatory runs. Results in `LEADERBOARD.md` up to v0.1 are **exploratory**:
they shaped the questions below and are not evidence for them. Any change to this plan after confirmatory runs
start is logged at the bottom with a date and a reason.

## Questions

**Q1. Does pretraining slow decay?** As a function of days since a decoder last saw labeled data, does a
pretrained model lose R² more slowly than the same architecture trained from scratch on the same held-in data?

**Q2. Does unlabeled data from the new day recover the loss?** For each decoder, how much of the drop from the
same-day level is recovered by adaptation that uses only the new day's unlabeled neural data?

**Q3. How much is just normalization?** How much of any stability advantage in Q1 or Q2 remains after every
decoder is given per-session z-scoring from the same unlabeled context?

## Data

- FALCON H1 (human, 7-D kinematics), M1 (monkey, 16-channel EMG), M2 (monkey, finger kinematics), public
  calibration files only. Held-out evaluation data on EvalAI is not used.
- Multi-session EEG datasets with recording dates, added under the same protocols (selection recorded below
  before any model is run on them).
- Every result carries a pretraining-overlap label from braindecode's registry
  (`braindecode.models.pretraining.check_pretraining_overlap`): clean, same collection, or in corpus. Results on
  in-corpus data are reported but excluded from the confirmatory Q1 comparison.

## Protocols

As implemented in `decay_bench/protocols.py` at the commit that adds this file:

- **B (primary): cross-session matrix.** Train on one held-in day, score every other calibration recording.
- **A (secondary): later sessions.** Train on all held-in days, score each held-out recording.
- Each test recording is split at the first trial start at or after its middle bin. The first part is
  unlabeled context, the second is scored with variance-weighted R² on FALCON evaluation bins.

## Conditions

For each model family (from-scratch baselines, pretrained models):

1. no adaptation, training-set normalization;
2. per-session z-scoring from the unlabeled context (the normalization control);
3. the model's own unsupervised adaptation, if it has one (for example SPINT's context trials, or
   test-time adaptation on the context), on top of 2.

## Outcomes

- **Primary:** decay slope, the least-squares slope of R² against log2(1 + days), fitted per training day
  (protocol B) with R² clipped at −1. Reported per dataset.
- **Secondary:** (a) mean R² in the 1–3, 4–7, 8–14, 15–30 and 31+ day bins; (b) days until R² falls below 80%
  of the same-day R² (linear interpolation on bin means; "not reached" if it never does); (c) share of collapsed
  pairs (R² < −1).
- A decoder with more than half of its pairs collapsed is reported as collapsed, without a slope.

## Seeds and statistics

- At least **10 seeds** per trained condition (training seed; data splits are fixed by the protocol).
  Deterministic decoders (ridge) use one run.
- Uncertainty: paired bootstrap (10,000 resamples) over training days, with seeds averaged within a training
  day, so comparisons are paired on the same training and test recordings.
- **Confirmatory comparisons** (one family per question, Holm correction within each family):
  - Q1: pretrained minus from-scratch, primary outcome, conditions 1 and 2 separately.
  - Q2: condition 3 minus condition 1, primary outcome and the 15–30 day bin.
  - Q3: the Q1 difference under condition 2 compared with condition 1.
- A difference counts as **supported** if its 95% CI (Holm-adjusted) excludes zero **and** its magnitude is at
  least 0.02 R² per doubling of days. Smaller differences with CIs excluding zero are reported as "detectable but
  small". Everything else is reported as inconclusive, not as "no effect", unless the CI lies within ±0.02.
- Results are reported per dataset. A conclusion is stated as general only if it holds on at least two of
  H1, M1 and M2.

## Things that will not be done

- No model selection or early stopping on held-out recordings. Training length is fixed in advance per model
  (or chosen on held-in minival only).
- No dropping of datasets, sessions or seeds after seeing results. Failed runs are re-run with the same seed
  and reported if they fail again.

## Deviations

(none yet)
