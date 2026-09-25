import numpy as np
import pandas as pd
import pytest
from ubs_recurrence.data import LABELS,PREDICTION,validate_submission,labels
from ubs_recurrence.evaluation import metrics
from ubs_recurrence.features import normalize


def test_metric_uses_all_eight_classes():
    y=np.arange(8)
    assert metrics(y,np.eye(8))["macro_f1"]==1
    p=np.zeros((8,8));p[:,0]=1
    assert metrics(y,p)["macro_f1"]==pytest.approx((2/9)/8)
    assert metrics(y,p)["accuracy"]==.125


def test_submission_contract():
    sample=pd.DataFrame({"client_id":["a","b"],PREDICTION:["none","none"]})
    assert validate_submission(sample.iloc[::-1],sample)
    for bad in [sample.iloc[:1],pd.concat([sample,sample.iloc[:1]]),sample.assign(**{PREDICTION:["fake","none"]}),sample.assign(client_id=["a","c"]),sample.rename(columns={PREDICTION:"wrong"})]:
        with pytest.raises(ValueError):validate_submission(bad,sample)


def test_holdout_requires_explicit_access():
    with pytest.raises(ValueError,match="holdout"):labels("valid")


def test_normalization_is_deterministic():
    assert normalize("CLOUD  Backup #182") == "cloud backup"
    assert normalize(normalize("Pay cloud BACKUP 123")) == normalize("Pay cloud BACKUP 123")


def test_probabilities_rejected_if_invalid():
    with pytest.raises(ValueError):metrics(np.arange(8),np.ones((8,8)))
