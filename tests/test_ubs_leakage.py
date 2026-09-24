"""Regressions for client isolation, causal features and the actual UBS runner."""

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transaction_forecasting.evaluation.official import LABELS, PREDICTION, TARGET
from transaction_forecasting.ubs.data import (
    load_ubs_data,
    read_labels,
    read_transactions,
    split_training_clients,
    validate_submission,
)
from transaction_forecasting.ubs.features import (
    ClientFeatureBuilder,
    build_client_documents,
    build_recurrence_streams,
)
from transaction_forecasting.ubs.models import CatBoostClientModel, ClientTextLogistic


def histories(prefix="train", copies=3):
    labels = pd.DataFrame(
        [
            {"client_id": f"{prefix}-{i}-{j}", "cutoff_date": "2026-01-01", TARGET: label}
            for i, label in enumerate(LABELS)
            for j in range(copies)
        ]
    )
    transactions = pd.DataFrame(
        [
            {
                "client_id": row.client_id,
                "timestamp": pd.Timestamp(f"2025-{month:02}-01", tz="UTC"),
                "amount": 10.0 + i,
                "currency": "chf",
                "description": f"observed merchant {getattr(row, TARGET)}",
                "direction": "out",
                "fee": 0.0,
                "mcc": "5734",
                "type": "card_payment",
            }
            for i, row in enumerate(labels.itertuples(index=False))
            for month in (10, 11, 12)
        ]
    )
    return transactions, labels


@pytest.fixture
def partitions(tmp_path):
    for split, copies in (("train", 4), ("valid", 1), ("test", 1)):
        transactions, labels = histories(split, copies)
        transactions.to_json(
            tmp_path / f"{split}_transactions.jsonl",
            orient="records",
            lines=True,
            date_format="iso",
        )
        if split != "test":
            labels.to_csv(tmp_path / f"{split}_labels.csv", index=False)
        else:
            labels[["client_id"]].iloc[::-1].assign(**{PREDICTION: ""}).to_csv(
                tmp_path / "sample_submission.csv", index=False
            )
    return tmp_path


@pytest.mark.parametrize("left,right", [("train", "valid"), ("train", "test"), ("valid", "test")])
def test_loader_rejects_every_client_overlap(partitions, left, right):
    path = partitions / f"{right}_transactions.jsonl"
    frame = read_transactions(path)
    frame["client_id"] = frame["client_id"].replace({f"{right}-0-0": f"{left}-0-0"})
    frame.to_json(path, orient="records", lines=True, date_format="iso")
    with pytest.raises(ValueError, match="Client leakage"):
        load_ubs_data(partitions)


@pytest.mark.parametrize("invalid", ["duplicate", "blank", "extra_column"])
def test_loader_checks_sample_before_model_fitting(partitions, invalid):
    path = partitions / "sample_submission.csv"
    sample = pd.read_csv(path, keep_default_na=False)
    if invalid == "duplicate":
        sample = pd.concat([sample, sample.iloc[:1]], ignore_index=True)
    elif invalid == "blank":
        sample.loc[0, "client_id"] = " "
    else:
        sample["unexpected"] = 1
    sample.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_ubs_data(partitions)


@pytest.mark.parametrize(
    "timestamp,allowed",
    [
        ("2025-12-31T23:59:59.999999Z", True),
        ("2026-01-01T00:30:00+01:00", True),
        ("2026-01-01T01:00:00+01:00", False),
        ("2025-12-31T23:30:00-01:00", False),
        (None, False),
    ],
)
def test_loader_enforces_cutoff_in_utc(tmp_path, timestamp, allowed):
    transactions, _ = histories(copies=1)
    row = transactions.iloc[0].to_dict()
    row["timestamp"] = timestamp
    path = tmp_path / "transactions.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    if allowed:
        assert len(read_transactions(path)) == 1
    else:
        with pytest.raises(ValueError):
            read_transactions(path)


def test_loader_preserves_literal_ids_and_none(tmp_path):
    transactions, labels = histories(copies=1)
    ids = {old: new for old, new in zip(labels.client_id[:2], ["001", "NA"], strict=True)}
    transactions["client_id"] = transactions.client_id.replace(ids)
    labels["client_id"] = labels.client_id.replace(ids)
    transactions.to_json(tmp_path / "tx.jsonl", orient="records", lines=True, date_format="iso")
    labels.to_csv(tmp_path / "labels.csv", index=False)
    assert {"001", "NA"}.issubset(set(read_transactions(tmp_path / "tx.jsonl").client_id))
    actual = read_labels(tmp_path / "labels.csv")
    assert actual.client_id.tolist() == labels.client_id.tolist()
    assert actual[TARGET].tolist() == labels[TARGET].tolist()


@pytest.mark.parametrize("invalid_time", [pd.NaT, pd.Timestamp("2026-01-01", tz="UTC")])
def test_direct_feature_entries_reject_unknown_or_future_times(invalid_time):
    transactions, labels = histories(copies=1)
    builder = ClientFeatureBuilder().fit(transactions, labels)
    transactions.loc[0, "timestamp"] = invalid_time
    for operation in (
        lambda: ClientFeatureBuilder().fit(transactions, labels),
        lambda: builder.transform(transactions),
        lambda: build_client_documents(transactions, labels.client_id),
        lambda: build_recurrence_streams(transactions),
    ):
        with pytest.raises(ValueError):
            operation()


def test_cross_fitted_features_cannot_see_own_label():
    transactions, labels = histories()
    client = labels.client_id.iloc[0]
    builder = ClientFeatureBuilder()
    features = builder.fit_transform(transactions, labels)
    changed = labels.copy()
    changed.loc[0, TARGET] = "none"
    changed_features = ClientFeatureBuilder().fit_transform(transactions, changed)
    pd.testing.assert_series_equal(features.loc[client], changed_features.loc[client])
    assert not features.filter(like="family_").equals(
        builder.transform(transactions).filter(like="family_")
    )
    # Inference retains a full-train mapping, not the final fold's mapping.
    pd.testing.assert_frame_equal(
        builder.description_lift_,
        ClientFeatureBuilder().fit(transactions, labels).description_lift_,
    )
    assert "client_id" not in features and TARGET not in features


def test_no_recurrence_split_has_same_schema_and_zero_evidence():
    transactions, labels = histories(copies=1)
    builder = ClientFeatureBuilder().fit(transactions, labels)
    expected = builder.transform(transactions)
    single_events = transactions.drop_duplicates("client_id")
    actual = builder.transform(single_events)
    assert actual.columns.tolist() == expected.columns.tolist()
    assert (actual.filter(like="family_") == 0).all().all()
    assert actual["repeated_description_count"].eq(0).all()


def test_lift_smoothing_does_not_invent_evidence_for_unseen_classes():
    transactions, labels = histories()
    labels[TARGET] = ["cloud", "gym"] * (len(labels) // 2)
    builder = ClientFeatureBuilder().fit(transactions, labels)
    assert (builder.description_lift_.loc[:, list(LABELS[2:])] == 0).all().all()


def test_amounts_and_streams_keep_currency_and_direction_separate():
    transactions, labels = histories(copies=1)
    original = transactions.iloc[:3].copy()
    dollars = original.assign(currency="usd", amount=1000.0)
    incoming = original.assign(direction="in", amount=2000.0)
    combined = pd.concat([original, dollars, incoming], ignore_index=True)
    streams = build_recurrence_streams(combined)
    assert len(streams) == 3
    assert streams.amount_cv.eq(0).all()
    features = ClientFeatureBuilder().fit(combined, labels.iloc[:1]).transform(combined)
    assert "amount_sum" not in features and "fee_sum" not in features
    assert not any(column.endswith("typical_amount") for column in features)
    assert features.iloc[0]["amount_usd_sum"] == 3000.0


def model_inputs():
    clients = pd.Index([f"client-{i}" for i in range(8)], name="client_id")
    features = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0], "empty": np.nan}, index=clients
    )
    documents = pd.Series(["cloud service"] * 4 + ["gym membership"] * 4, index=clients)
    target = pd.Series(["cloud"] * 4 + ["gym"] * 4, index=clients)
    return features, documents, target


def test_logistic_fits_only_train_and_aligns_documents_and_labels():
    features, documents, target = model_inputs()
    model = ClientTextLogistic().fit(features, documents.iloc[::-1], target.iloc[::-1])
    reference = ClientTextLogistic().fit(features, documents, target)
    np.testing.assert_allclose(model.model.coef_, reference.model.coef_)
    np.testing.assert_allclose(
        model.predict_proba(features, documents.iloc[::-1]),
        reference.predict_proba(features, documents),
    )
    statistics = model.imputer.statistics_.copy()
    means = model.scaler.mean_.copy()
    vocabulary = model.vectorizer.vocabulary_.copy()
    held = features.fillna(0).add(10000)
    probabilities = model.predict_proba(held, documents + " unseenfutureword")
    np.testing.assert_equal(model.imputer.statistics_, statistics)
    np.testing.assert_equal(model.scaler.mean_, means)
    assert model.vectorizer.vocabulary_ == vocabulary
    assert "unseenfutureword" not in vocabulary
    assert probabilities.shape == (8, 8)
    assert (probabilities[:, 2:] == 0).all()
    assert len(model.feature_importance()) == len(model.numeric_columns_) + len(vocabulary)
    with pytest.raises(ValueError, match="IDs must match"):
        model.predict_proba(features, documents.iloc[:-1])


def test_catboost_missing_classes_and_train_only_imputation():
    features, _, target = model_inputs()
    model = CatBoostClientModel(balanced=True, iterations=3, depth=2).fit(
        features, target.iloc[::-1]
    )
    medians = model.medians_.copy()
    probabilities = model.predict_proba(features.fillna(0).add(10000))
    pd.testing.assert_series_equal(model.medians_, medians)
    assert probabilities.shape == (8, 8)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities[:, 2:] == 0).all()


def test_inference_does_not_learn_categories_or_label_associations():
    transactions, labels = histories()
    builder = ClientFeatureBuilder().fit(transactions, labels)
    levels = {column: values.copy() for column, values in builder.categorical_levels_.items()}
    lift = builder.description_lift_.copy()
    held = transactions.assign(description="novel token", mcc="never seen", currency="eur")
    features = builder.transform(held)
    assert builder.categorical_levels_ == levels
    pd.testing.assert_frame_equal(builder.description_lift_, lift)
    assert (features.filter(like="family_") == 0).all().all()
    assert "currency_eur_count" not in features


def test_internal_split_is_stratified_reproducible_and_disjoint():
    _, labels = histories(copies=4)
    fit_ids, selection_ids = split_training_clients(labels)
    reordered_fit, reordered_selection = split_training_clients(labels.iloc[::-1])
    assert fit_ids.equals(reordered_fit) and selection_ids.equals(reordered_selection)
    assert not set(fit_ids) & set(selection_ids)
    assert set(fit_ids) | set(selection_ids) == set(labels.client_id)
    assert len(fit_ids) == 24 and len(selection_ids) == 8
    assert labels.set_index("client_id").loc[selection_ids, TARGET].value_counts().eq(1).all()


def test_runner_generates_and_reloads_valid_submission(partitions, monkeypatch):
    runner = Path(__file__).resolve().parents[1] / "scripts" / "run_ubs_baseline.py"
    output = partitions / "results"
    from transaction_forecasting.ubs import evaluation

    training_labels = read_labels(partitions / "train_labels.csv")
    expected_fit_ids, _ = split_training_clients(training_labels)
    expected_all_ids = set(training_labels.client_id)

    def allowed_fit_ids():
        if (output / "selected_model.json").exists():
            return expected_all_ids
        return set(expected_fit_ids)

    fit_memberships = []
    original_fit = ClientFeatureBuilder.fit

    def guarded_fit(self, transactions, labels):
        assert labels.client_id.str.startswith("train-").all()
        assert set(labels.client_id) <= allowed_fit_ids()
        fit_memberships.append(set(labels.client_id))
        return original_fit(self, transactions, labels)

    monkeypatch.setattr(ClientFeatureBuilder, "fit", guarded_fit)
    for model_class in (ClientTextLogistic, CatBoostClientModel):
        model_fit = model_class.fit

        def guarded_model_fit(self, features, *args, _fit=model_fit):
            assert features.index.str.startswith("train-").all()
            assert set(features.index) <= allowed_fit_ids()
            return _fit(self, features, *args)

        monkeypatch.setattr(model_class, "fit", guarded_model_fit)

    official_scores = []
    original_evaluate = evaluation.evaluate_predictions

    def guarded_evaluate(target, prediction):
        if target.index.str.startswith("valid-").any():
            assert (output / "selected_model.json").is_file()
            official_scores.append(target.index.tolist())
        return original_evaluate(target, prediction)

    monkeypatch.setattr(evaluation, "evaluate_predictions", guarded_evaluate)
    config = partitions / "config.toml"
    config.write_text(
        f"""[project]
random_seed = 42
[data]
directory = "{partitions.as_posix()}"
cutoff = "2026-01-01"
target = "{TARGET}"
[features]
recent_windows = [7, 14, 30, 60, 90, 180]
tfidf_max_features = 50
[logistic]
c_values = [1.0]
class_weights = ["none"]
[catboost]
iterations = 3
depth = 2
learning_rate = 0.05
[outputs]
metrics_directory = "{output.as_posix()}"
submission = "{(output / 'submission.csv').as_posix()}"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", [str(runner), "--config", str(config)])
    runpy.run_path(str(runner), run_name="__main__")
    data = load_ubs_data(partitions)
    submission = pd.read_csv(output / "submission.csv", dtype=str, keep_default_na=False)
    validate_submission(submission, data.sample_submission, data.test_transactions)
    protocol = json.loads((output / "selection_protocol.json").read_text())
    assert len(protocol["heuristic_trials"]) == 19
    assert len(protocol["ensemble_trials"]) == 10
    assert "not_independent" in protocol["evaluation_role"]
    assert protocol["official_validation_used_for_selection"] is False
    assert protocol["official_validation_used_for_refit"] is False
    assert fit_memberships[0] == set(protocol["fit_clients"])
    assert not fit_memberships[0] & set(protocol["selection_clients"])
    assert len(official_scores) == 1
    recipe = json.loads((output / "selected_model.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert summary["final_refit_clients"] == 32
    assert summary["internal_fit_clients"] == 24
    assert summary["internal_selection_clients"] == 8
    assert summary["valid_clients"] == 8
    assert summary["final_refit_uses_train_and_valid"] is False
    report = (output / "report.md").read_text()
    assert "not an independent estimate" in report
    # Mutating every official label and every placeholder cannot change selection or test.
    labels_path = partitions / "valid_labels.csv"
    changed = pd.read_csv(labels_path)
    changed[TARGET] = changed[TARGET].map(dict(zip(LABELS, (*LABELS[1:], LABELS[0]), strict=True)))
    changed.to_csv(labels_path, index=False)
    sample_path = partitions / "sample_submission.csv"
    sample = pd.read_csv(sample_path, keep_default_na=False)
    sample[PREDICTION] = "placeholder_is_not_a_target"
    sample.to_csv(sample_path, index=False)
    old_output = output
    output = partitions / "changed_validation_results"
    config.write_text(config.read_text().replace(old_output.as_posix(), output.as_posix()))
    runpy.run_path(str(runner), run_name="__main__")
    assert json.loads((output / "selected_model.json").read_text()) == recipe
    pd.testing.assert_frame_equal(
        submission, pd.read_csv(output / "submission.csv", dtype=str, keep_default_na=False)
    )
    assert len(official_scores) == 2
    # A different configured cutoff must never silently use January's labels.
    config.write_text(config.read_text().replace("2026-01-01", "2025-01-01"))
    with pytest.raises(ValueError, match="official cutoff"):
        runpy.run_path(str(runner), run_name="__main__")
