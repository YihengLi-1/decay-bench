# Pretraining overlap of entries with FALCON

A pretrained model's decay curve on a FALCON recording means something different if that recording, or another
session of the same subject, was in its pretraining corpus. Each pretrained entry gets a mark per dataset:

- **[H]** held out: neither the recording nor its subject is in the pretraining corpus;
- **[S]** same subject: other sessions of the test subject are in the corpus (the test days themselves are not);
- **[I]** in corpus: the test recordings themselves were pretrained on;
- **[?]** not checkable from public material.

## NDT3 (Ye et al. 2025; github.com/joel99/ndt3 at 877f9f5)

Checked from the released configs, not from the data. The v5 pretraining configs exclude every alias matching
`falcon.*` (`context_general_bci/config/exp/v5/_default.yaml:131`, commented "DO NOT TOUCH - TEST SET
FINE-TUNING"). That excludes the FALCON-formatted copies only. The question is whether the same subjects come in
through the labs' raw archives, which are pretrained on under other aliases.

| Checkpoint | Corpus (config) | H1 (Pitt human) | M1 (Rouse lab, monkey L) | M2 (Chestek lab, monkey N) |
|---|---|---|---|---|
| `base_45m_200h` | `v5/_200h.yaml`: churchland, gallego, dyer_co, delay, miller, flint, Rouse_A, Rouse_B, chase_Rocky, chase_Nigel-2019, perich | [H] no Pitt human data | [H] Rouse monkeys A, B only | [H] |
| `big_350m_2kh` | `v5/_2kh.yaml`: the above with all of `rouse.*` and `chase.*`, plus schwartz, chicago, pitt_broad | **[?]** `pitt_broad.*` is Pitt's private human archive; whether it holds the H1 participant is not visible in the code | [H] Rouse monkeys in the registry are P, Q, A, B | [H]* |

\* `schwartz/MonkeyN` (Schwartz lab) shares the name "Monkey N" with FALCON M2's Chestek-lab animal; these are
presumably different animals. A Chestek-lab monkey N finger dataset (`mender_fingerctx`) is registered but is not
in either pretraining list.

**Status:** pending author confirmation, mainly the H1 question for `big_350m_2kh`. Until it is confirmed, NDT3
350M results on H1 are reported with a [?] mark and left out of the confirmatory Q1 comparison
(`ANALYSIS_PLAN.md`).
