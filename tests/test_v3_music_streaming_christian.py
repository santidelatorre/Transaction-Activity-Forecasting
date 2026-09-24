"""Focused regression checks for client isolation and conservative overrides."""

import pandas as pd
import pytest

from transaction_forecasting.ubs.data import LABELS


def load_experiment():
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1] / "scripts/experiments/v3_music_streaming_christian.py"
    )
    spec = spec_from_file_location("music_streaming_experiment", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base_scores():
    clients = pd.Index(["c1", "c2", "c3"], name="client_id")
    base = pd.DataFrame(0.01, index=clients, columns=LABELS)
    base.loc["c1", "cloud"] = 0.7
    base.loc["c2", "none"] = 0.7
    base.loc["c3", "music"] = 0.7
    specialist = pd.DataFrame(
        {"music": [0.9, 0.9, 0.1], "streaming": [0.05, 0.05, 0.85], "other": [0.05, 0.05, 0.05]},
        index=clients,
    )
    return base, specialist


def test_override_preserves_none_and_existing_specialist_predictions():
    module = load_experiment()
    base, specialist = base_scores()
    prediction = module.apply_override(base, specialist, threshold=0.85, margin=0.2)
    assert prediction.to_dict() == {"c1": "music", "c2": "none", "c3": "music"}
    with pytest.raises(ValueError, match="align"):
        module.apply_override(base, specialist.iloc[::-1], threshold=0.85, margin=0.2)


def test_specialist_refuses_training_clients_at_prediction():
    module = load_experiment()
    clients = pd.Index(["a", "b", "c", "d", "e", "f"], name="client_id")
    documents = pd.Series(
        ["music service", "music tune", "streaming music", "streaming show", "music shop", "gym"],
        index=clients,
    )
    recurrence = pd.DataFrame({"events": [2, 3, 4, 2, 1, 0]}, index=clients)
    target = pd.Series(["music", "music", "streaming", "streaming", "none", "gym"], index=clients)
    model = module.Specialist(recurrence=False).fit(documents, recurrence, target)
    with pytest.raises(ValueError, match="seen"):
        model.predict(documents, recurrence)
