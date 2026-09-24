"""Small, comparable UBS description/merchant-proxy ablations on official train/valid."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from transaction_forecasting.evaluation.official import classification_metrics
from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.features import ClientFeatureBuilder, build_client_documents
from transaction_forecasting.ubs.text_v2 import (
    MerchantFeatureBuilder,
    client_documents,
    normalize_description,
)


def _numeric(
    train: pd.DataFrame, valid: pd.DataFrame
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_array = scaler.fit_transform(imputer.fit_transform(train))
    valid_array = scaler.transform(imputer.transform(valid[train.columns]))
    return sparse.csr_matrix(train_array), sparse.csr_matrix(valid_array)


def _vectorizer(kind: str, config: dict) -> TfidfVectorizer:
    common = dict(
        min_df=int(config["min_df"]),
        max_df=float(config["max_df"]),
        sublinear_tf=True,
        strip_accents="unicode",
    )
    if kind == "char":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(int(config["char_ngram_min"]), int(config["char_ngram_max"])),
            max_features=int(config["char_max_features"]),
            **common,
        )
    return TfidfVectorizer(
        ngram_range=(1, 1) if kind == "word1" else (1, 2),
        max_features=int(config["max_features"]),
        **common,
    )


def _score(target: pd.Series, prediction: np.ndarray) -> dict:
    return classification_metrics(target, pd.Series(prediction, index=target.index))


def _v1_reference(directory: Path, clients: pd.Index, target: pd.Series) -> dict | None:
    path = directory / "validation_error_analysis.csv"
    if not path.is_file():
        return None
    errors = pd.read_csv(path, dtype={"client_id": str}).set_index("client_id")
    if not errors.index.is_unique or set(errors.index) != set(clients):
        raise ValueError("V1 validation predictions have different clients")
    errors = errors.reindex(clients)
    if not errors["actual"].eq(target).all():
        raise ValueError("V1 validation labels do not match this dataset")
    return _score(target, errors["model_prediction"].to_numpy())


def _top_terms(transactions: pd.DataFrame, target: pd.Series, vectorizer: TfidfVectorizer) -> dict:
    """Client-level term lift without bigrams spanning transaction boundaries."""
    descriptions = transactions["description"].map(normalize_description)
    transaction_terms = vectorizer.transform(descriptions).tocsr()
    positions = pd.Categorical(transactions["client_id"], categories=target.index).codes
    if (positions < 0).any():
        raise ValueError("Term analysis has clients without train labels")
    membership = sparse.csr_matrix(
        (np.ones(len(positions)), (positions, np.arange(len(positions)))),
        shape=(len(target), len(positions)),
    )
    presence = (membership @ transaction_terms).tocsr()
    presence.data = np.ones_like(presence.data)
    names = vectorizer.get_feature_names_out()
    terms: dict[str, list[dict]] = {}
    for label in LABELS:
        mask = target.to_numpy() == label
        inside = np.asarray(presence[mask].sum(axis=0)).ravel()
        outside = np.asarray(presence[~mask].sum(axis=0)).ravel()
        lift = ((inside + 1) / (mask.sum() + 2)) / ((outside + 1) / ((~mask).sum() + 2))
        eligible = np.flatnonzero(inside >= 20)
        best = eligible[np.argsort(lift[eligible])[-8:][::-1]]
        terms[label] = [
            {
                "term": str(names[index]),
                "train_clients": int(inside[index]),
                "lift": round(float(lift[index]), 3),
            }
            for index in best
        ]
    return terms


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = frame.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend(
        "| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ubs_text_v2.toml")
    args = parser.parse_args()
    with Path(args.config).open("rb") as stream:
        settings = tomllib.load(stream)
    project, model_config, text_config = (settings["project"], settings["model"], settings["tfidf"])
    seed = int(project["random_seed"])
    np.random.seed(seed)
    data = load_ubs_data(project["data_directory"])
    builder = ClientFeatureBuilder().fit(data.train_transactions, data.train_labels)
    x_train = builder.transform(data.train_transactions)
    x_valid = builder.transform(data.valid_transactions)
    y_train = data.train_labels.set_index("client_id")[TARGET_COLUMN].reindex(x_train.index)
    y_valid = data.valid_labels.set_index("client_id")[TARGET_COLUMN].reindex(x_valid.index)
    if y_train.isna().any() or y_valid.isna().any():
        raise ValueError("Missing target after client alignment")

    merchant = MerchantFeatureBuilder().fit(data.train_transactions, data.train_labels)
    merchant_train = merchant.transform(data.train_transactions, training=True).reindex(
        x_train.index
    )
    merchant_valid = merchant.transform(data.valid_transactions).reindex(x_valid.index)
    numeric_train, numeric_valid = _numeric(x_train, x_valid)
    merchant_num_train, merchant_num_valid = _numeric(
        pd.concat([x_train, merchant_train], axis=1),
        pd.concat([x_valid, merchant_valid], axis=1),
    )
    raw_train = build_client_documents(data.train_transactions, x_train.index)
    raw_valid = build_client_documents(data.valid_transactions, x_valid.index)
    clean_train = client_documents(data.train_transactions, x_train.index)
    clean_valid = client_documents(data.valid_transactions, x_valid.index)
    texts: dict[str, tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]] = {}
    vectorizers: dict[str, TfidfVectorizer] = {}
    for name, kind, train_docs, valid_docs in (
        ("v1_word", "word2", raw_train, raw_valid),
        ("normalized_unigram", "word1", clean_train, clean_valid),
        ("normalized_word12", "word2", clean_train, clean_valid),
        ("normalized_char", "char", clean_train, clean_valid),
    ):
        vectorizer = _vectorizer(kind, text_config)
        train_text = vectorizer.fit_transform(train_docs).tocsr()
        valid_text = vectorizer.transform(valid_docs).tocsr()
        texts[name] = (train_text, valid_text, vectorizer.get_feature_names_out())
        vectorizers[name] = vectorizer

    reference = _v1_reference(Path(project["v1_metrics_directory"]), x_valid.index, y_valid)
    results: dict[str, dict] = {}
    if reference is not None:
        results["v1_selected"] = {
            "metrics": reference,
            "train_seconds": None,
            "feature_count": None,
            "vocabulary_size": None,
        }

    def run(
        name: str,
        numeric_pair: tuple[sparse.csr_matrix, sparse.csr_matrix],
        text_name: str | None = None,
    ) -> None:
        train_matrix, valid_matrix = numeric_pair
        vocabulary_size = 0
        if text_name is not None:
            train_text, valid_text, names = texts[text_name]
            train_matrix = sparse.hstack([train_matrix, train_text], format="csr")
            valid_matrix = sparse.hstack([valid_matrix, valid_text], format="csr")
            vocabulary_size = len(names)
        started = perf_counter()
        classifier = LogisticRegression(
            C=float(model_config["c"]),
            class_weight=None
            if model_config["class_weight"] == "none"
            else model_config["class_weight"],
            max_iter=int(model_config["max_iter"]),
            random_state=seed,
            solver="lbfgs",
            tol=1e-5,
        ).fit(train_matrix, y_train)
        seconds = perf_counter() - started
        predicted = classifier.predict(valid_matrix)
        results[name] = {
            "metrics": _score(y_valid, predicted),
            "train_seconds": round(seconds, 3),
            "feature_count": int(train_matrix.shape[1]),
            "vocabulary_size": vocabulary_size,
        }

    run("numeric_control", (numeric_train, numeric_valid))
    run("v1_word_component", (numeric_train, numeric_valid), "v1_word")
    run("normalized_unigram", (numeric_train, numeric_valid), "normalized_unigram")
    run("normalized_word12", (numeric_train, numeric_valid), "normalized_word12")
    run("normalized_char", (numeric_train, numeric_valid), "normalized_char")
    run("merchant_proxy", (merchant_num_train, merchant_num_valid))
    best_text = max(
        ("normalized_word12", "normalized_char"),
        key=lambda name: results[name]["metrics"]["macro_f1"],
    )
    run("text_plus_merchant", (merchant_num_train, merchant_num_valid), best_text)

    base = reference or results["v1_word_component"]["metrics"]
    rows = []
    class_rows = []
    for name, result in results.items():
        metrics = result["metrics"]
        rows.append(
            {
                "variant": name,
                "macro_f1": round(metrics["macro_f1"], 6),
                "delta_v1": round(metrics["macro_f1"] - base["macro_f1"], 6),
                "accuracy": round(metrics["accuracy"], 6),
                "features": result["feature_count"],
                "vocabulary": result["vocabulary_size"],
                "train_s": result["train_seconds"],
            }
        )
        for label in LABELS:
            current = metrics["per_class"][label]
            prior = base["per_class"][label]
            class_rows.append(
                {
                    "variant": name,
                    "class": label,
                    "support": current["support"],
                    "f1": round(current["f1"], 4),
                    "delta_v1_f1": round(current["f1"] - prior["f1"], 4),
                }
            )
    terms = _top_terms(data.train_transactions, y_train, vectorizers["normalized_word12"])
    directory = Path(project["output_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": settings,
        "reference": "v1_selected" if reference else "v1_word_component",
        "combined_text": best_text,
        "results": results,
        "discriminative_terms": terms,
        "train_clients": len(x_train),
        "valid_clients": len(x_valid),
    }
    (directory / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    comparison = pd.DataFrame(rows)
    per_class = pd.DataFrame(class_rows)
    diversity_rows = []
    for partition, features, target, documents in (
        ("train", merchant_train, y_train, clean_train),
        ("valid", merchant_valid, y_valid, clean_valid),
    ):
        diagnostic = features.assign(target=target, has_text=documents.str.len().gt(0))
        for label, group in diagnostic.groupby("target"):
            diversity_rows.append(
                {
                    "partition": partition,
                    "class": label,
                    "clients": len(group),
                    "text_coverage": round(float(group["has_text"].mean()), 3),
                    "unique_median": round(float(group["merchant_unique"].median()), 2),
                    "entropy_median": round(float(group["merchant_entropy"].median()), 2),
                    "top_share_median": round(float(group["merchant_top_share"].median()), 3),
                }
            )
    diversity = pd.DataFrame(diversity_rows)
    comparison.to_csv(directory / "comparison.csv", index=False)
    per_class.to_csv(directory / "per_class.csv", index=False)
    diversity.to_csv(directory / "merchant_diversity.csv", index=False)
    for name, result in results.items():
        pd.DataFrame(result["metrics"]["confusion_true_by_predicted"]).T.reindex(
            index=LABELS, columns=LABELS
        ).to_csv(directory / f"confusion_{name}.csv")
    term_lines = [
        f"- **{label}:** "
        + ", ".join(
            f"{item['term']} ({item['lift']:.2f}×; {item['train_clients']} clientes)"
            for item in terms[label]
        )
        for label in LABELS
    ]
    report = f"""# UBS V2: texto y aproximación de merchant

La columna `merchant` no existe en el contrato UBS. Se usa `description` normalizada
como aproximación, sin atribuir una familia a cada transacción. Todas las filas
usan train/valid oficiales y el mismo corte de V1 (`2026-01-01`).

## Hipótesis y configuraciones

La normalización puede reducir referencias variables. Los n-grams de caracteres
pueden tolerar variantes de escritura. La diversidad y recurrencia de descripciones
puede añadir señales que el TF-IDF agregado pierde. La regresión logística usa
semilla {seed}, C={model_config["c"]}, pesos={model_config["class_weight"]},
`min_df={text_config["min_df"]}`, `max_df={text_config["max_df"]}` y vocabularios
limitados a {text_config["max_features"]} palabras o {text_config["char_max_features"]} caracteres.
`v1_word_component` reproduce la representación de la regresión logística V1,
con el C del experimento; `v1_selected` procede del runner V1 si existe su salida.

## Comparación en validación

{_markdown_table(comparison)}

La combinación usa `{best_text}`, elegido entre dos representaciones por su
Macro-F1 en esta misma validación. Su mejora es exploratoria: necesita otro
split o test ciego antes de concluir que generaliza.

## F1 por clase (delta frente a la referencia V1)

{_markdown_table(per_class)}

El soporte es el número de clientes reales de cada clase. Las matrices de
confusión completas están en el directorio de resultados.

## Concentración y diversidad de descripciones

{_markdown_table(diversity)}

`unique_median` cuenta descripciones normalizadas distintas por cliente;
`top_share_median` mide qué fracción de sus movimientos usa la más común.
Estas cifras son descriptivas, no identifican comercios reales.

## Términos discriminativos por clase

Lift de presencia en clientes de la clase frente al resto, estimado únicamente
con train y un mínimo de veinte clientes de la clase por término:

{chr(10).join(term_lines)}

## Conclusión y límites

Tomar como candidato sólo una variante que supere a `v1_selected` en Macro-F1
y sin deterioro grave de clases minoritarias. `description` puede contener
palabras literales de la familia del generador sintético; verificar robustez
antes de integrarla. Vocabularios y asociaciones se ajustaron con train; para
features supervisadas de entrenamiento se resta la contribución del propio
cliente. Los comercios de alta cardinalidad pueden generar asociaciones frágiles.

Ejecutado con `python scripts/run_ubs_text_v2.py --config configs/ubs_text_v2.toml`.
"""
    (directory / "report.md").write_text(report, encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"Report: {directory / 'report.md'}")


if __name__ == "__main__":
    main()
