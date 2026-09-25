import json
import numpy as np
import pandas as pd
import pytest
import ubs_recurrence.data as data


def test_target_alignment_is_by_id_not_file_order(tmp_path,monkeypatch):
    raw=tmp_path/"data/raw";raw.mkdir(parents=True)
    pd.DataFrame({"client_id":["B","A"],"cutoff_date":["2026-01-01"]*2,data.TARGET:["none","cloud"]}).to_csv(raw/"train_labels.csv",index=False)
    monkeypatch.setattr(data,"ROOT",tmp_path)
    np.testing.assert_array_equal(data.aligned_target(["A","B"]),[0,7])
    with pytest.raises(ValueError):data.aligned_target(["A","A"])
    with pytest.raises(ValueError):data.aligned_target(["A","C"])


def test_illegal_and_duplicate_labels_are_rejected(tmp_path,monkeypatch):
    raw=tmp_path/"data/raw";raw.mkdir(parents=True);monkeypatch.setattr(data,"ROOT",tmp_path)
    for ids,labels in [(["A","B"],["cloud","unknown"]),(["A","A"],["cloud","none"])]:
        pd.DataFrame({"client_id":ids,"cutoff_date":["2026-01-01"]*2,data.TARGET:labels}).to_csv(raw/"train_labels.csv",index=False)
        with pytest.raises(ValueError):data.labels()


def test_post_cutoff_transaction_is_rejected(tmp_path,monkeypatch):
    raw=tmp_path/"data/raw";raw.mkdir(parents=True);monkeypatch.setattr(data,"ROOT",tmp_path)
    event={"client_id":"A","timestamp":"2026-01-01T00:00:00Z","amount":10,"currency":"chf","direction":"out","type":"card_payment","mcc":"5732","description":"cloud access","fee":0}
    (raw/"train_transactions.jsonl").write_text(json.dumps(event)+"\n")
    with pytest.raises(ValueError,match="cutoff"):data.transactions("train",use_cache=False)
