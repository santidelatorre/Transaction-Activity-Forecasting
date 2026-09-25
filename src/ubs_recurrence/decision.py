"""Decision biases fitted exclusively on training OOF predictions."""

import numpy as np
from scipy.special import softmax
from sklearn.metrics import f1_score


def apply_bias(p, bias):
    return softmax(
        np.log(np.maximum(np.asarray(p, dtype=float), 1e-12)) + np.asarray(bias), axis=1
    )


def optimize_bias(probability_sets, y):
    """Small deterministic coordinate search; its training score is not validation."""
    logps = [np.log(np.maximum(p, 1e-12)) for p in probability_sets]

    def objective(b):
        return float(
            np.mean(
                [
                    f1_score(
                        y,
                        np.argmax(p + b, axis=1),
                        labels=np.arange(8),
                        average="macro",
                        zero_division=0,
                    )
                    for p in logps
                ]
            )
        )

    bias = np.zeros(8)
    history = [{"step": "initial", "score": objective(bias), "bias": bias.tolist()}]
    for step in [0.2, 0.1, 0.05]:
        for _ in range(3):
            improved = False
            for k in range(8):
                best = objective(bias)
                chosen = bias.copy()
                for delta in [-step, step]:
                    candidate = bias.copy()
                    candidate[k] += delta
                    candidate -= candidate.mean()
                    if abs(candidate).max() > 1:
                        continue
                    score = objective(candidate)
                    if score > best + 1e-9:
                        best = score
                        chosen = candidate
                        improved = True
                bias = chosen
            history.append(
                {"step": step, "score": objective(bias), "bias": bias.tolist()}
            )
            if not improved:
                break
    return bias, history
