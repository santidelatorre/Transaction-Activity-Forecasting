"""Reproducible feature-only split audit and training-label forensics.

Never reads valid_labels.csv. Cached parquet contains transactions only.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import subprocess
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "outputs/audit"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "data/cache"
CACHE.mkdir(parents=True, exist_ok=True)
CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z]+", " ", s.lower())).strip()


def main():
    expected = json.loads((ROOT / "reports/data_manifest.json").read_text())
    manifest = {}
    for path in sorted(RAW.iterdir()):
        manifest[path.name] = {"bytes": path.stat().st_size,
                               "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest()}
    if manifest != expected:
        raise ValueError("Raw files differ from pinned manifest; run prepare_data.py to verify inputs")
    frames, info, idsets, signatures = {}, {}, {}, {}
    schema = {"client_id", "timestamp", "amount", "currency", "direction", "type", "mcc", "description", "fee"}
    for split in ["train", "valid", "test", "unlabeled_pretrain"]:
        df = pd.read_json(RAW / f"{split}_transactions.jsonl", lines=True, dtype={"mcc": str})
        assert set(df) == schema, (split, df.columns)
        df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)
        assert df.timestamp.max() < CUTOFF, (split, df.timestamp.max())
        assert df.client_id.notna().all()
        df = df.sort_values(["client_id", "timestamp"], kind="stable").reset_index(drop=True)
        df.to_parquet(CACHE / f"{split}.parquet", index=False)
        df["normalized_description"] = df.description.map(norm)
        idsets[split] = set(df.client_id)
        group = df.groupby("client_id")
        count = group.size()
        duration = (group.timestamp.max() - group.timestamp.min()).dt.total_seconds()/86400
        signatures[split] = group.apply(lambda g: hashlib.sha256(pd.util.hash_pandas_object(g, index=False).values.tobytes()).hexdigest(), include_groups=False)
        info[split] = {
            "transactions": len(df), "clients": len(count),
            "transactions_per_client": count.describe().to_dict(),
            "history_days": duration.describe().to_dict(),
            "time_range": [str(df.timestamp.min()), str(df.timestamp.max())],
            "missing": df.isna().sum().to_dict(), "duplicate_transactions": int(df.duplicated().sum()),
            "duplicate_histories": int(signatures[split].duplicated().sum()),
            "amount": df.amount.describe(percentiles=[.01,.1,.5,.9,.99]).to_dict(),
            "fee": df.fee.describe().to_dict(),
            "unique_descriptions": df.description.nunique(),
            "unique_normalized_descriptions": df.normalized_description.nunique(),
        }
        for col in ["type", "mcc", "currency", "direction"]:
            info[split][col] = df[col].value_counts().to_dict()
        for field in ["hour", "minute", "second", "dayofweek", "day", "month"]:
            info[split]["timestamp_"+field] = getattr(df.timestamp.dt, field).value_counts().sort_index().to_dict()
        df.description.value_counts().rename("count").to_csv(OUT / f"{split}_descriptions.csv")
        df.groupby(["description", "mcc", "type", "direction"]).amount.agg(["count","mean","std","min","max"]).to_csv(OUT / f"{split}_description_structure.csv")
        count.rename("transaction_count").to_csv(OUT / f"{split}_client_counts.csv")
        print(split, len(df), len(count), df.description.nunique(), flush=True)
        frames[split] = df if split == "train" else df[["client_id","description","normalized_description"]]
    overlaps = {}
    names = list(idsets)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            overlaps[f"{a}:{b}"] = {"ids":len(idsets[a]&idsets[b]), "histories":len(set(signatures[a])&set(signatures[b]))}
    assert all(x["ids"] == 0 for x in overlaps.values()), overlaps
    info["overlaps"] = overlaps
    train_vocab = set(frames["train"].description)
    info["unseen_description_rates"] = {s: float((~df.description.isin(train_vocab)).mean()) for s,df in frames.items()}
    labels = pd.read_csv(RAW / "train_labels.csv").set_index("client_id")
    assert labels.index.is_unique and set(labels.index) == idsets["train"]
    assert labels.cutoff_date.unique().tolist() == ["2026-01-01"]
    sample = pd.read_csv(RAW / "sample_submission.csv")
    assert sample.client_id.is_unique and set(sample.client_id) == idsets["test"]
    target = "target_next_recurring_merchant"
    info["train_labels"] = labels[target].value_counts().to_dict()
    tr = frames["train"].join(labels[target], on="client_id", validate="many_to_one")
    for col in ["description", "normalized_description", "mcc", "type", "direction", "currency"]:
        for unit,rows in [("client", tr.drop_duplicates(["client_id",col])), ("event",tr)]:
            table = pd.crosstab(rows[col],rows[target])
            table.to_csv(OUT / f"target_by_{col}_{unit}_counts.csv")
            table.div(table.sum(axis=1),axis=0).to_csv(OUT / f"target_by_{col}_{unit}_prob.csv")
    tokens = tr[["client_id","normalized_description",target]].copy()
    tokens["token"] = tokens.normalized_description.str.split()
    tokens = tokens.explode("token").drop_duplicates(["client_id","token"])
    pd.crosstab(tokens.token,tokens[target]).to_csv(OUT / "target_by_token_counts.csv")
    numeric = tr.groupby("client_id").agg(n=("amount","size"),amount_mean=("amount","mean"),amount_std=("amount","std"),first=("timestamp","min"),last=("timestamp","max"))
    numeric["recency"] = (CUTOFF-numeric["last"]).dt.total_seconds()/86400
    numeric["duration"] = (numeric["last"]-numeric["first"]).dt.total_seconds()/86400
    numeric.join(labels[target]).groupby(target).describe().to_csv(OUT / "client_statistics_by_target.csv")
    (OUT / "summary.json").write_text(json.dumps(info,indent=2,default=str))
    print(json.dumps({k: v for k,v in info.items() if k not in names},indent=2))


if __name__ == "__main__":
    main()
