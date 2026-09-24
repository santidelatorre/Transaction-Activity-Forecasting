from __future__ import annotations

import pandas as pd
import pytest

from transaction_forecasting.ubs.text_v2 import (
    MerchantFeatureBuilder,
    client_documents,
    normalize_description,
)


def test_normalization_keeps_merchant_words_and_masks_variable_tokens() -> None:
    assert normalize_description("  CAFÉ-Shop #A123456 2025-12-31 CHF 42.50!! ") == (
        "cafe shop id_token date_token chf num_token"
    )
    assert normalize_description(None) == ""
    assert normalize_description(float("nan")) == ""


def test_documents_preserve_repeated_descriptions_and_client_order() -> None:
    frame = pd.DataFrame(
        {"client_id": ["A", "A", "B"], "description": ["Cloud 12", "Cloud 13", "Gym!"]}
    )
    docs = client_documents(frame, pd.Index(["B", "A"], name="client_id"))
    assert docs.tolist() == ["gym", "cloud num_token cloud num_token"]


def test_merchant_class_features_exclude_own_training_label() -> None:
    frame = pd.DataFrame(
        {
            "client_id": ["A", "A", "B", "C"],
            "description": ["Cloud 101", "Cloud 102", "Cloud 103", "Gym 99"],
        }
    )
    labels = pd.DataFrame(
        {
            "client_id": ["A", "B", "C"],
            "target_next_recurring_merchant": ["cloud", "cloud", "gym"],
        }
    )
    builder = MerchantFeatureBuilder().fit(frame, labels)
    train = builder.transform(frame, training=True)
    valid = builder.transform(pd.DataFrame({"client_id": ["V"], "description": ["Cloud 200"]}))
    assert train.loc["A", "merchant_unique"] == 1
    assert train.loc["A", "merchant_top_share"] == 1
    assert train.loc["A", "merchant_repeated_share"] == 1
    assert train.loc["A", "merchant_class_cloud_max"] < valid.loc["V", "merchant_class_cloud_max"]
    assert train.loc["C", "merchant_train_client_frequency_max"] == 0
    with pytest.raises(ValueError, match="unknown client"):
        builder.transform(
            pd.DataFrame({"client_id": ["V"], "description": ["Cloud 200"]}), training=True
        )
