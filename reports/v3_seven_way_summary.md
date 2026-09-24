# V3 seven-way synthesis — decision summary

**Decision: ONE MORE TARGETED ROUND. Do not build V3.1 yet.**

Keep V3-A as the research base and V2 as the fallback. No addition establishes
a transferable gain. Full evidence: [seven-way synthesis](v3_seven_way_synthesis.md).

## Best results, with the necessary distinctions

- **Best supported clean result:** V3-A, **0.424111097737 VALID Macro-F1**,
  **0.459793826869 TRAIN OOF**. Jaime's actual branch,
  `origin/experiment/v3a-v2-blend`, selects **alpha=1**, exactly A. Its incremental
  OOF and VALID deltas versus A are both **zero**. Only that alpha is scored on VALID.
- **Highest frozen, TRAIN-selected new candidate:** gym protection,
  **0.427115231 VALID**, **+0.003004133** versus A. However, gate-CV OOF delta is
  **−0.000898517**. The positive selection rescore (+.000408496) is not that CV
  estimate. Five VALID changes recover three gym cases. Do not promote it.
- **Highest predefined diagnostic control:** fixed confidence routing,
  **0.441966427217 VALID**, but **0.444519429432 OOF**, below A. It was not
  parameter-tuned on VALID; promoting it now would be retrospective selection.
  The learned router fails both comparisons.

“Clean” means no identified direct evaluation-label fit or new VALID parameter
selection; VALID has been repeatedly used.

## What actually survived scrutiny

| Experiment | OOF delta versus A | VALID delta versus A | Conclusion |
| --- | --- | --- | --- |
| NONE gate | +.025225 | −.067038 | Reject; 502 NONE predictions, every class F1 lower |
| Learned router | −.002195 | −.006817 | Reject; 12 corrections, 13 regressions |
| Soft family mapping | +.020823 | −.099737 | Reject; 576 NONE predictions |
| Music/streaming | +.007926 | .000000 | Zero VALID overrides; no added signal at the frozen gate |
| Calibration | +.009746 | −.006875 | Accuracy rises while gym/cloud and Macro-F1 fall |
| Gym gate CV | −.000899 | +.003004 | Five-client diagnostic; failed TRAIN support |
| Jaime blend | .000000 | .000000 | Alpha=1 retains A |

**Most trustworthy improvement remains the original hard family identity:**
A−V2 is **+.050910281 OOF / +.032561642 VALID**, with all five original folds
positive. The new soft posterior is not a semantic replacement: its seven
positive probabilities implicitly expose the omitted NONE posterior and mix
coverage with evidence strength.

**Most useful complementary signal remains V2.** It uniquely corrects 90 A
errors on VALID, notably gym/software. This does not imply a global blend or
learned gate can identify those clients at inference. Jaime's zero V2 weight
and the failed learned router are evidence against those tested methods.

Among new experiments, the failed family posterior supplies the most novel
answers: **70 clients both V2 and A miss**, including 17 positive-family cases
and 53 NONE. Investigate its transfer failure; do not integrate it. A/V2's
diagnostic oracle reaches .519729 F1; its loose routing bound is only .546154.

## Biggest bottleneck and next work

A makes **312 wrong-positive-family decisions**, versus **227 positive↔NONE
errors**, on VALID. Streaming/music/software remain weak. At the same time,
OOF-to-full-fit score transfer is unreliable: true NONE prevalence is nearly
unchanged, but A's NONE prediction rate moves from 15.2% OOF to 28.6% VALID.
Several corrections learn to fix the former and damage the latter.

Run at most three sharply defined tests, in order:

1. **P0:** TRAIN-only map-size transfer experiment. Compare ordinary full-map
   inference with inference averaged over the corresponding inner-fit maps,
   holding the classifier fixed, across prespecified training sizes. Test A
   and the rejected soft-posterior arm to separate fit-size from cohort effects.
2. **P1:** a small, label-blind, TRAIN-only reviewed semantic alias map, keeping
   A's feature dimensions/model fixed. Test actual semantic identity, not more
   target co-occurrence or generic string cleanup.
3. **P2:** a client-grouped shared family scorer trained against the official
   positive-client family target, preserving the baseline's NONE decisions.

The report fixes thresholds, safe folds and stop conditions. This path targets
shared family errors and score transfer. The ~.59 leaderboard claim is unverified.

**Do not** stack the seven settings, lower thresholds using VALID errors, promote
the highest VALID control, restore full temporal V3, generate a submission, or
merge experimental branches. Four second-layer studies also reuse base OOF
without full end-to-end nesting; their gate-CV estimates need that qualification.

All principal OOF/VALID results were reproduced and independently rescored;
router/gym used unchanged modules through a portable adapter. A 10,000-draw
paired bootstrap gives gym−A `[0,.006817]` and fixed-router−A
`[.000192,.035544]`; neither corrects for selection or overrides adverse OOF.
The audit found no metric, evaluator or dataset-label changes. No experiment
was merged and no submission was generated. Full-suite pytest and Ruff checks
are recorded in the full report.
