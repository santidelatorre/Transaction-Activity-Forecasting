# Classwise calibration protocol (frozen before execution)

Base: `df9fe4135ff812add0e2f6e0733706050b813cc2`, V3-A probabilities only.
No additional classifier, V2/V3 blend, temporal model, or temperature search.

Reuse and verify the original five client-stratified TRAIN OOF probability files
(seed 42). Report V2 on the identical clients. For calibration outer fold k,
fit only on the other four OOF folds. Select the parameters by leave-one-fold-out
CV within those four folds, recomputing the frequency direction without each
inner holdout. Fit the final correction on all TRAIN OOF predictions, selecting
its coefficients by the analogous five-fold calibration CV. Freeze its JSON
before reading VALID labels. Evaluate exactly that correction on VALID once.

Correction: `q[c] = p[c] exp(b[c]) / sum_j p[j] exp(b[j])`.
This has the same argmax as additive log-probability offsets, with log(0)=-inf.
It is a decision correction, not evidence of calibrated probability confidence.

For each fitting set, compute `d = log((true_count+5)/(raw_predicted_count+5))`,
subtract its class mean, and shrink uniformly so max(abs(d)) <= .25.
Set `b = clip(alpha*d + beta*(gym-none) + gamma*(music-streaming), -.25, .25)`.
The notation `(gym-none)` denotes a vector with +1 for gym, -1 for none, zero
elsewhere; similarly for music-streaming. Grid: alpha in {0,.25,.5,.75,1}; beta
and gamma in {0,-.075,.075,-.15,.15}: 125 deterministic candidates, including
identity. Thus eight bounded offsets arise from three searched coefficients
and a predetermined frequency estimate, not eight independently optimized biases.

Selection objective: pooled inner held-out Macro-F1 minus
`.02 * mean_inner_folds(mean_classes((b/.25)^2))`. Require at least .001 gain
over identity in this regularized objective. Reject any candidate that loses
more than .02 F1 in one class, more than .005 in more than two classes, or fails
to preserve/improve at least four classes, relative to raw on the inner held-out
rows. Deterministic enumeration breaks ties; identity comes first. These guards
apply to selection data and do not guarantee outer/VALID per-class improvements.

Report pooled outer cross-fitted Macro-F1, accuracy, every class F1, fold-specific
coefficients and offsets, distributions, and correct/incorrect transitions.
Report the final all-OOF fit's inner selection score only as a selection diagnostic,
never as the unbiased TRAIN headline. Recommend promotion only if TRAIN supports
a useful stable gain without sacrificing multiple classes; VALID is descriptive
and cannot cause a different parameter choice or an additional experiment.

Limitation: this nests the calibration layer, not the original classifier. Each
base OOF row excludes its own client's label, but other OOF folds' base models
were trained with the calibration outer holdout's clients. Indirect dependence
therefore remains across base OOF folds. Do not call this fully nested end-to-end
CV. Reused official VALID and a V3-A architecture informed by earlier VALID
research are also not an untouched generalization test.
