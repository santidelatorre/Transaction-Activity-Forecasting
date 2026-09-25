# Final candidate freeze (before second holdout batch)

The 0.80 objective remains unmet. No score in the auxiliary historical task is
a challenge result. The only official validation access so far was the frozen
v1 batch: control 0.177187 macro-F1, robustness ensemble 0.587318, calibrated
variant 0.586551. That batch established the deployment shift and did not
confirm the apparent development benefit of class-specific bias optimization.

Subsequent changes were evaluated on training-client CV with fixed corruption
stress sets. The key accepted change is amount-matched refund context, backed
by explicit refund/fee/time ablations. The historical auxiliary expert, sparse
two-observation candidates, and fee/time-only changes did not improve the
selected stress protocol enough to retain. Three CV seeds (42, 17, 2026) were
run and all are included in the final ensemble, without selecting the best.

Final predictor:
- 75%: equal average of three seeded compact family rankers, each with a
  separate binary none detector.
- 25%: equal average of the three frozen complementary robustness rankers
  (hard-family LightGBM, soft-price-family LightGBM, soft-price-family XGBoost).
- Final decision: argmax of the averaged normalized scores; no fitted biases.
- All supervised fitting uses the original 2,000 training clients only.
- All three noise views of a client stay in its training fold during CV.
- Price profiles use only the disjoint unlabeled pretraining histories.

The fixed complementary blend improves mean valid-like/test-like train OOF
macro-F1 by 0.009026, exceeding the predeclared 0.005 retention threshold.
Train OOF: 0.655305 original; 0.637367 valid-like; 0.613157 test-like.
These are exploratory model-selection estimates. Synthetic corruption stress
does not exactly reconstruct deployment corruption. Official validation is
the external check, and the second check follows this frozen specification.

Next: rebuild every selected feature from pinned raw JSONL, fit every selected
model from scratch, evaluate once, then independently repeat the raw-data
build and training. Compare all probabilities and predictions, validate the
test submission contract, run tests, and document limitations. Final model
training will not incorporate official validation labels.
