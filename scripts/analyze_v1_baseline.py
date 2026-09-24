"""Generate aggregate V1 evidence and paired V1/V2 reports after reproduction."""

# Markdown report templates intentionally contain long table rows.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Resolve this checkout explicitly; the audit also contains isolated historical copies.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transaction_forecasting.ubs.data import (
    CUTOFF,
    LABELS,
    PREDICTION_COLUMN,
    TARGET_COLUMN,
    load_ubs_data,
    validate_submission,
)
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.features import ClientFeatureBuilder

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/metrics/v1_benchmark"
REPORTS = ROOT / "reports"
V1 = "0199a8b7c8b2c790a9d3447b156f2088708c2864"
V2 = "96409b7940a991fbda5235b85ffeb40652a4087b"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(frame):
    frame = frame.copy()
    for col in frame.select_dtypes(include="number"):
        frame[col] = frame[col].map(lambda x: f"{x:.6f}" if isinstance(x, float) else str(x))
    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    lines += [
        "| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(lines)


def save_report(name, title, sections):
    text = f"# {title}\n\n" + "\n\n".join(f"## {heading}\n\n{body}" for heading, body in sections)
    (REPORTS / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def main():
    REPORTS.mkdir(exist_ok=True)
    s1 = read_json(OUT / "v1/summary.json")
    s2 = read_json(OUT / "v2/summary.json")
    data = load_ubs_data(ROOT / "data/raw/ubs_2026")
    target = data.valid_labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    e1 = pd.read_csv(OUT / "v1/validation_error_analysis.csv", dtype={"client_id": str})
    e2 = pd.read_csv(OUT / "v2/validation_predictions.csv", dtype=str, keep_default_na=False)
    for frame in (e1, e2):
        if not frame.client_id.is_unique or set(frame.client_id) != set(target.index):
            raise ValueError("Validation prediction IDs must match labels exactly")
    p1 = e1.set_index("client_id").model_prediction.reindex(target.index)
    p2 = e2.set_index("client_id")[PREDICTION_COLUMN].reindex(target.index)
    if not e1.set_index("client_id").actual.reindex(target.index).equals(target.rename("actual")):
        raise ValueError("V1 saved labels differ from official labels")
    m1, m2 = evaluate_predictions(target, p1), evaluate_predictions(target, p2)
    assert np.isclose(m1["macro_f1"], s1["best_macro_f1"], atol=1e-12)
    assert np.isclose(m2["macro_f1"], s2["metrics"]["macro_f1"], atol=1e-12)
    correct1, correct2 = p1.eq(target), p2.eq(target)
    paired = {
        "both_correct": int((correct1 & correct2).sum()),
        "only_v1_correct": int((correct1 & ~correct2).sum()),
        "only_v2_correct": int((~correct1 & correct2).sum()),
        "both_wrong": int((~correct1 & ~correct2).sum()),
        "same_prediction": int(p1.eq(p2).sum()),
        "different_prediction": int(p1.ne(p2).sum()),
    }
    class_rows = []
    for label in LABELS:
        mask = target.eq(label)
        a, b = m1["per_class"][label], m2["per_class"][label]
        class_rows.append(
            {
                "class": label,
                "support": int(mask.sum()),
                "V1 predicted": int(p1.eq(label).sum()),
                "V2 predicted": int(p2.eq(label).sum()),
                "V1 precision": a["precision"],
                "V2 precision": b["precision"],
                "delta precision": b["precision"] - a["precision"],
                "V1 recall": a["recall"],
                "V2 recall": b["recall"],
                "delta recall": b["recall"] - a["recall"],
                "V1 F1": a["f1-score"],
                "V2 F1": b["f1-score"],
                "delta F1": b["f1-score"] - a["f1-score"],
                "corrected": int((mask & ~correct1 & correct2).sum()),
                "introduced": int((mask & correct1 & ~correct2).sum()),
            }
        )
    classes = pd.DataFrame(class_rows)
    overall = pd.DataFrame(
        [
            {"metric": metric, "V1": m1[metric], "V2": m2[metric], "delta": m2[metric] - m1[metric]}
            for metric in ("macro_f1", "accuracy")
        ]
    )
    cm1 = pd.DataFrame(m1["confusion_matrix"], index=LABELS, columns=LABELS)
    cm2 = pd.DataFrame(m2["confusion_matrix"], index=LABELS, columns=LABELS)
    errors = []
    for actual in LABELS:
        for predicted in LABELS:
            if actual != predicted:
                count = int(cm1.loc[actual, predicted])
                errors.append(
                    {
                        "actual": actual,
                        "predicted": predicted,
                        "errors": count,
                        "share_of_class_errors": count
                        / max(1, cm1.loc[actual].sum() - cm1.loc[actual, actual]),
                    }
                )
    errors = pd.DataFrame(errors).sort_values("errors", ascending=False).head(12)
    submissions = []
    test_frames = []
    for name in ("v1", "v2"):
        path = OUT / name / "submission.csv"
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        validate_submission(frame, data.sample_submission, data.test_transactions)
        test_frames.append(frame.set_index("client_id")[PREDICTION_COLUMN])
        submissions.append(
            {
                "version": name,
                "rows": len(frame),
                "unique_ids": frame.client_id.nunique(),
                "status": "PASS",
                "sha256": digest(path),
            }
        )
    test_same = int(test_frames[0].eq(test_frames[1]).sum())
    inventory = []
    paths = sorted(
        set((ROOT / "outputs/predictions").glob("*.csv"))
        | {
            OUT / "v1/submission.csv",
            OUT / "v2/submission.csv",
            OUT / "v2/validation_predictions.csv",
            ROOT / "data/raw/ubs_2026/sample_submission.csv",
        }
    )
    for path in paths:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        ids = set(frame.client_id) if "client_id" in frame else set()
        kind = "other experimental output"
        if path.name == "sample_submission.csv":
            kind = "C: template, not truth"
        elif ids == set(target.index):
            kind = "A: validation predictions"
        elif ids == set(data.sample_submission.client_id):
            kind = "B: test predictions; labels unavailable"
        inventory.append(
            {"path": path.relative_to(ROOT).as_posix(), "kind": kind, "sha256": digest(path)}
        )
        if kind.startswith("B:"):
            try:
                validate_submission(frame, data.sample_submission, data.test_transactions)
                inventory[-1]["schema_validation"] = "PASS"
            except ValueError as error:
                inventory[-1]["schema_validation"] = str(error)
        else:
            inventory[-1]["schema_validation"] = "NOT A TEST SUBMISSION"
    builder = ClientFeatureBuilder().fit(data.train_transactions, data.train_labels)
    features = builder.transform(data.valid_transactions).reindex(target.index)
    feature_names = features.columns.tolist()
    diagnostic = features[
        [
            "n_transactions",
            "history_days",
            "days_since_last_transaction",
            "repeated_description_count",
            "stream_interval_cv_mean",
            "best_recurrence_score",
        ]
    ].copy()
    diagnostic["correct"] = correct1
    groups = diagnostic.groupby("correct").median().reset_index()
    per_class_groups = (
        diagnostic.assign(actual=target).groupby(["actual", "correct"]).median().reset_index()
    )
    cutoff_audit = []
    for name in ("train", "valid", "test"):
        tx = getattr(data, f"{name}_transactions")
        cutoff_audit.append(
            {
                "split": name,
                "transactions": len(tx),
                "clients": tx.client_id.nunique(),
                "at_or_after_cutoff": int(tx.timestamp.ge(CUTOFF).sum()),
                "exact_duplicates": int(tx.duplicated().sum()),
                "repeated_client_timestamp_rows": int(
                    tx.duplicated(["client_id", "timestamp"]).sum()
                ),
            }
        )
    none_transitions = {
        "positive_to_none": int((p1.ne("none") & p2.eq("none")).sum()),
        "none_to_positive": int((p1.eq("none") & p2.ne("none")).sum()),
        "true_none_corrected": int((target.eq("none") & ~correct1 & correct2).sum()),
        "true_none_regressed": int((target.eq("none") & correct1 & ~correct2).sum()),
    }
    versions = {
        package: version(package)
        for package in ("numpy", "pandas", "scikit-learn", "catboost", "matplotlib")
    }
    versions["python"] = platform.python_version()
    evidence = {
        "v1_commit": V1,
        "v2_commit": V2,
        "v1": m1,
        "v2": m2,
        "paired": paired,
        "per_class": class_rows,
        "none_transitions": none_transitions,
        "test_same_predictions": test_same,
        "submissions": submissions,
        "csv_inventory": inventory,
        "data_checks": cutoff_audit,
        "versions": versions,
        "feature_names": feature_names,
        "data_sha256": read_json(OUT / "v1/manifest.json")["data_sha256"],
    }
    evidence["source_sha256"] = {
        name: {
            path.relative_to(OUT / f"source_{name}").as_posix(): digest(path)
            for folder in ("src", "scripts", "configs")
            for path in sorted((OUT / f"source_{name}" / folder).rglob("*"))
            if path.suffix in (".py", ".toml")
        }
        for name in ("v1", "v2")
    }
    (REPORTS / "v1_benchmark_evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    paired_frame = pd.DataFrame(
        {"actual": target, "v1": p1, "v2": p2, "v1_correct": correct1, "v2_correct": correct2}
    )
    paired_frame.to_csv(OUT / "paired_validation.csv", index_label="client_id")
    figure_dirs = [REPORTS / "figures/v1_baseline", REPORTS / "figures/v1_vs_v2"]
    for path in figure_dirs:
        path.mkdir(parents=True, exist_ok=True)
    for name, cm, folder in (("v1", cm1, figure_dirs[0]), ("v2", cm2, figure_dirs[1])):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
        for ax, values, title in (
            (axes[0], cm.to_numpy(), "Counts"),
            (axes[1], cm.div(cm.sum(axis=1), axis=0).to_numpy(), "Fraction of true class"),
        ):
            im = ax.imshow(values, cmap="Blues")
            ax.set_xticks(range(8), LABELS, rotation=45, ha="right")
            ax.set_yticks(range(8), LABELS)
            ax.set(xlabel="Predicted", ylabel="Actual", title=f"{name.upper()} — {title}")
            fig.colorbar(im, ax=ax, shrink=0.7)
        fig.savefig(folder / f"{name}_confusion.png", dpi=130)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    x = np.arange(8)
    axes[0, 0].bar(x - 0.2, classes["V1 F1"], 0.4, label="V1")
    axes[0, 0].bar(x + 0.2, classes["V2 F1"], 0.4, label="V2")
    axes[0, 0].legend()
    axes[0, 0].set_title("F1 by class")
    axes[0, 1].bar(x, classes["delta F1"])
    axes[0, 1].axhline(0, color="black", linewidth=0.6)
    axes[0, 1].set_title("V2 minus V1 F1")
    for offset, col, label in (
        (-0.25, "support", "True"),
        (0, "V1 predicted", "V1"),
        (0.25, "V2 predicted", "V2"),
    ):
        axes[1, 0].bar(x + offset, classes[col], 0.25, label=label)
    axes[1, 0].legend()
    axes[1, 0].set_title("Validation class counts")
    axes[1, 1].bar(x - 0.2, classes.corrected, 0.4, label="Corrected")
    axes[1, 1].bar(x + 0.2, classes.introduced, 0.4, label="Introduced")
    axes[1, 1].legend()
    axes[1, 1].set_title("Errors changed by V2")
    for ax in axes.flat:
        ax.set_xticks(x, LABELS, rotation=35, ha="right")
    fig.savefig(figure_dirs[1] / "comparison.png", dpi=130)
    plt.close(fig)
    write_reports(
        s1,
        s2,
        m1,
        m2,
        classes,
        overall,
        cm1,
        cm2,
        errors,
        paired,
        none_transitions,
        submissions,
        inventory,
        cutoff_audit,
        groups,
        per_class_groups,
        feature_names,
        versions,
        test_same,
    )
    print(json.dumps({"v1": m1["macro_f1"], "v2": m2["macro_f1"], "paired": paired}, indent=2))


def write_reports(
    s1,
    s2,
    m1,
    m2,
    classes,
    overall,
    cm1,
    cm2,
    errors,
    paired,
    none_transitions,
    submissions,
    inventory,
    cutoff_audit,
    groups,
    per_class_groups,
    feature_names,
    versions,
    test_same,
):
    experiments = pd.read_csv(OUT / "v1/experiments.csv")
    commands = """```powershell
.\\.venv\\Scripts\\python.exe scripts/reproduce_v1_benchmark.py --version v1
.\\.venv\\Scripts\\python.exe scripts/reproduce_v1_benchmark.py --version v2
.\\.venv\\Scripts\\python.exe scripts/analyze_v1_baseline.py
.\\.venv\\Scripts\\python.exe -m pytest -q
```
Los scripts fijan commits, crean worktrees detached bajo outputs ignorados y copian los seis archivos de datos.
No cambian la rama activa ni sobrescriben las submissions originales. Las rutas de salida son la única
modificación de configuración V1. Logs, predicciones por cliente y manifests quedan en
`outputs/metrics/v1_benchmark/`. Reejecutar sustituye únicamente esos outputs de reproducción.
"""
    architecture = """`ubs.data.load_ubs_data` carga train/valid/test y sus contratos. `ClientFeatureBuilder.fit`
aprende vocabularios y asociación supervisada descripción-familia con train. `transform` agrega cada split
a una fila por cliente; el ID es índice de alineación, no una feature numérica. Se comparan dummy,
heurístico, seis regresiones logísticas, dos CatBoost y un ensemble seleccionado. Se elige el máximo
Macro-F1 (accuracy desempata) sobre valid. Finalmente se reconstruye el builder sobre 3.000 clientes
train+valid y se aplica la receta seleccionada a los 1.000 test. Las métricas proceden de antes del refit.
No se usa el archivo opcional de preentrenamiento sin etiquetas.

Fuentes: `scripts/run_ubs_baseline.py`, `src/transaction_forecasting/ubs/{data,features,models,evaluation}.py`.
"""
    feature_text = """El builder produce **218 columnas** en esta ejecución. El ganador heurístico utiliza solo
cuatro por familia (32 entradas); no utiliza directamente todas las 218 columnas ni TF-IDF.

| Bloque | Definición / información | Ausentes y limitaciones |
| --- | --- | --- |
| Volumen/dirección | n_transactions, n_out, n_in, out_share=n_out/n | Una fila por cliente observado; sin manejo de clientes sin movimientos |
| Importes y fees | sum/mean/median/std/min/max de amount; suma/media/ratio positivo de fee | Agregados generales mezclan monedas nominales; no son importes convertidos |
| Diversidad | número de MCC, descripciones, tipos, monedas | Identidad por descripción normalizada, no merchant real |
| Historia | history_days=max(t)-min(t); recencia=cutoff-max(t); transacciones/30 días | Duración se limita a 1 día para evitar división por cero |
| Ventanas | counts, outgoing y counts/window en 7/14/30/60/90/180 días | Ventanas fijadas en features.py; recent_windows de TOML no gobierna ese código |
| Calendario | shares por día/fin de semana; meses activos; media/std/CV mensual | Estadística mensual incluye meses con actividad, no todos los meses vacíos |
| Gaps | media/mediana/std entre eventos consecutivos por cliente | Mezcla comercios; timestamps repetidos generan gaps cero |
| Categorías | count/share por MCC, tipo, moneda y dirección | Vocabulario ajustado en train; categorías nuevas no generan columnas nuevas |
| Moneda | suma/media/mediana/std de amount por moneda | Convive con agregados generales sin convertir |
| Streams | grupos cliente+descripción con al menos dos eventos | Mezcla monedas/tipos bajo descripción; no deduplica timestamps para intervalos |
| Recurrencia | counts repetidos, regularidad, estabilidad de importe, score máximo/medio | Dos observaciones dan un intervalo con std poblacional cero: evidencia de regularidad débil |
| Periodicidad | número de streams en [4,10), [10,18), [18,45), [45,110), [110,400) días | Etiqueta annual abarca 110–400 días: no demuestra periodicidad anual |
| Familia | 9 columnas x 8 familias = 72: occurrences, descriptions, recency, frequency, regularity, typical_amount, amount_similarity, description_lift, recurrence_score | Asociación supervisada; filas train se codifican usando su propia etiqueta en los candidatos ML |
| Texto (candidatos LR) | TF-IDF word unigram/bigram, min_df=2, max_df=.98, sublinear_tf, hasta 2500 términos | Ajustado solo en train; no es parte del ganador heurístico |

En presencia de streams, el builder rellena NaN con cero. Inf se convierte a NaN al final.
Si no hay streams globalmente, no crea todas las columnas familiares: edge case de esquema.
El inventario exacto está en `v1_benchmark_evidence.json` y al final de este apartado.

**Ecuaciones.** Para stream s: d=mediana de intervalos, r=cutoff-último evento,
CV_t=std(intervalos)/max(d,1), R=1/(1+CV_t),
D=exp(-abs(r-d)/max(d,7)), CV_a=std(amount)/max(mean(amount),1e-9),
B=log(1+n)*R*D/(1+CV_a). El score no calcula explícitamente una probabilidad de evento en 90 días.

Para descripción u y clase c: lift=clip(log(((n_uc+1)/(N_c+2))/((n_u_notc+1)/(N_notc+2))),0,log(10)).
Se conserva solo la clase con lift máximo si lift>=log(1.5) y support>=2 clientes.
Por cliente/familia, recurrence_score suma B*lift; occurrences suma apariciones,
regularity y description_lift son máximos de streams con evidencia.

Logit de familia c = log(1+recurrence_score_c) + .15 log(1+occurrences_c)
+ .20 description_lift_c + .10 regularity_c. A none se añade bias; softmax(logit/T)
produce los valores usados en selección. No son probabilidades calibradas demostradas.

Inventario completo: """ + ", ".join(f"`{x}`" for x in feature_names)
    audit = """| Severidad | Evidencia / ubicación | Consecuencia | Estado y recomendación |
| --- | --- | --- | --- |
| Alta | models.RecurrenceHeuristic.tune recibe x_valid/y_valid; runner selecciona modelos/alpha sobre valid | Métrica de selección optimista; no es prueba independiente | Histórico preservado. Reservar un test nuevo o evaluación interna separada |
| Alta | ClientFeatureBuilder.fit usa etiquetas train; transform(train) reutiliza el mismo mapa | Candidatos LR/CatBoost reciben codificación influida por su propia etiqueta | No atribuir este mecanismo a filtración de etiquetas valid en el ganador heurístico; V2 elimina 72 columnas del ML |
| Media | data.py exige timestamps < cutoff e IDs disjuntos; ver tabla de controles | Reduce filtración temporal y entre clientes | Controles ejecutados; no prueba absoluta de ausencia de leakage |
| Media | Importes generales y streams mezclan monedas nominales | Scores de estabilidad no equivalen a magnitudes monetarias comparables | No corregido en reproducción; ya existen momentos por moneda |
| Media | features._add_recurrence_features retorna antes si streams.empty | Esquema incompleto en datos sin recurrencias; posible KeyError heurístico | No ocurre en datos oficiales de esta ejecución; agregar fallback en futuro |
| Media | Intervalos no deduplicados; dos eventos dan std=0 | Regularidad aparente con evidencia escasa | Riesgo de robustez, no prueba de leakage |
| Baja | cutoff/target/recent_windows declarados en config pero definidos por constantes del código | Config puede aparentar flexibilidad que no tiene | Documentar las constantes efectivas |
| Control | Vocabulario, imputer, scaler, TF-IDF ajustados con train; IDs solo índice | No se observó fit de estos componentes con valid/test | Inspección de código, no auditoría de todo futuro cambio |
| Control | Refit train+valid separado del scoring | Etiquetas valid autorizadas para modelo final test | Nunca puntuar refit sobre valid como holdout |

No se modificó el predictor para maquillar o corregir V1. No hay test oculto etiquetado local.
No se fabricaron targets históricos: la etiqueta por cliente de enero no etiqueta cada transacción.
"""
    protocol = """Split oficial: 2.000 clientes train, 1.000 valid, 1.000 test; cutoff UTC 2026-01-01;
horizonte objetivo 90 días. Macro-F1 es media simple de ocho F1, F1_c=2TP/(2TP+FP+FN),
zero_division=0. Alineación por client_id comprobada, no por posición accidental.
V1 seleccionó configuración en valid. V2 también fue seleccionada mediante experimentos sobre ese
valid. Comparación descriptiva justa en el mismo conjunto; no evaluación independiente ni estimación
garantizada del leaderboard. El CSV de test carece de etiquetas y no permite calcular accuracy/F1.
"""
    selection = """La reproducción ejecuta el procedimiento histórico, sin ampliar su búsqueda:
dummy mayoritario; 95 combinaciones del heurístico (19 biases -2..2.5 por .25 y 5 temperaturas);
6 LR (C=.3,1,3 x pesos none/balanced); 2 CatBoost (350 iteraciones, profundidad 6,
learning_rate=.05, seed=42, pesos normal/balanced); 10 mezclas del mejor ML con heurístico
(alpha=.05.. .50). Se registra el mejor ensemble como una fila, no todos los alpha.
La temperatura positiva no cambia el argmax del heurístico aislado; sí sus probabilidades en mezclas.
Desempate heurístico favorece bias cercano a cero y temperatura cercana a 1.
El ganador es recurrence_heuristic con none_bias=-1 y temperature=1.
Un bias negativo reduce none; no aumenta su sensibilidad.

Resultados históricos reproducidos de candidatos:\n\n""" + table(experiments)
    comparison_arch = """V2 fija CatBoost balanceado (300 iteraciones, profundidad 4, learning_rate=.05,
seed=42, cuatro threads) sobre **146 features de historia sin etiquetas**. Excluye las 72 family_*.
Combina .75 probabilidades CatBoost + .25 heurístico de periodicidad, bias=-1/T=1.
El mapa supervisado separado solo se usa para inferencia en clientes excluidos de fit.
`IntegratedV2Model.predict_components` rechaza solapamiento con clientes de entrenamiento.
Código inspeccionado en el worktree V2: `ubs/v2.py`, `ubs/temporal_features.py`, `scripts/run_ubs_v2.py`.
No se reintegraron ramas ni se hizo tuning nuevo.
"""
    runtime = (
        f"V1 runner completo: {s1['pipeline_seconds']:.2f} s. V2 fit+refit: {s2['seconds']:.2f} s. "
        + """Son costes end-to-end de trabajos diferentes: V1 incluye selección completa y V2 receta congelada.
Se ejecutaron concurrentemente, por lo que estos tiempos no constituyen un benchmark aislado de latencia.
Los tiempos por candidato V1 se incluyen en la tabla de experimentos.
"""
    )
    reproducibility = (
        "Python y librerías:\n\n```json\n"
        + json.dumps(versions, indent=2)
        + "\n```\n\n"
        + """Commits y hashes de entradas/submissions están en `v1_benchmark_evidence.json`.
El V2 remoto reporta otro runtime Python; por eso se comparan hashes además de métricas.
Se usa PYTHONPATH del worktree correspondiente para impedir importar accidentalmente otro predictor.
"""
    )
    tests = """V1/base: 37 tests pasaron en esta sesión. Primer intento: 34 pases y 3 errores de permisos
en el directorio temporal de Windows; repetición con acceso autorizado: 37 pases, un warning de caché.
Los 28 tests citados por el equipo son una cifra histórica, no el total del commit reproducido.
Resultados de V2, lint y pre-commit de esta auditoría se registran en `v1_benchmark_checks.md`.
"""
    submission_text = (
        table(pd.DataFrame(submissions))
        + f"\n\nTest: {test_same} predicciones coinciden; {1000-test_same} difieren. No se conoce cuál acierta.\n\n"
        + """Se validaron columnas exactas, IDs únicos, cobertura, orden sample, ausencia de nulos y vocabulario.
Submissions reproducidas: `outputs/metrics/v1_benchmark/v1/submission.csv` y `v2/submission.csv`.
No son prueba de que el CSV enviado al dashboard sea el mismo: no se recibió un adjunto CSV en esta tarea.
El informe remoto declara SHA V1 b67d364e207d297b1a4d5898f0c37b6f62289cf56808882f5bdcfeb6c5795b0f
y V2 da1314cfdea23ffea7756ff82db4e7d4efe77c6fcaf33cb0b66a689795f59adc; cotejar con tabla.
"""
    )
    evidence_sources = f"""V1: `{V1}`; V2: `{V2}`. El informe V2 remoto identifica explícitamente
0199a8b como baseline y describe 0.391549456 para la mezcla final. Ambas versiones se ejecutaron
desde worktrees detached del commit exacto. Los informes históricos orientaron la identificación;
las métricas de este informe se recalcularon de las predicciones frescas y etiquetas oficiales.

Inventario de CSV locales:\n\n""" + table(pd.DataFrame(inventory))
    main_result = table(overall)
    class_table = table(classes)
    cm_text = (
        "### V1\n\n"
        + table(cm1.rename_axis("actual / predicted").reset_index())
        + "\n\n### V2\n\n"
        + table(cm2.rename_axis("actual / predicted").reset_index())
    )
    none_text = (
        table(classes[classes["class"].eq("none")])
        + "\n\n"
        + table(pd.DataFrame([none_transitions]))
        + "\n\nnone es una clase real, no abstención. V1 la penaliza con bias=-1 pese a ser la más frecuente."
    )
    error_text = (
        table(errors)
        + "\n\nMedianas V1 correcto/incorrecto (descriptivas, no causales):\n\n"
        + table(groups)
        + "\n\nPor clase:\n\n"
        + table(per_class_groups)
    )
    findings = """V1 es un control reproducible e interpretable con buen recall relativo en algunas familias,
pero falla especialmente en none. La codificación supervisada in-sample afecta a los candidatos ML;
no invalida automáticamente la inferencia del heurístico sobre clientes valid excluidos del fit.
V2 gana globalmente en este valid reutilizado. Music retrocede; streaming apenas mejora en F1.
El informe de integración contiene ablations: history CatBoost .383950488 y blend .391549456;
el pequeño delta de mezcla .007598968 no prueba por sí solo generalización. No se reejecutaron esas
ablations en este encargo. La comparación final no identifica causalmente cuánto aporta cada cambio.
"""
    sections = [
        ("1. Executive summary", main_result + "\n\n" + findings),
        ("2. Scope and evidence sources", evidence_sources),
        (
            "3. Exact baseline identification",
            f"V1 `{V1}`. Config `configs/ubs_v1.toml`, seed=42. Ganador `{s1['best_model']}`. El código histórico se conserva sin cambios.",
        ),
        (
            "4. Dataset and prediction target",
            protocol
            + "\n\n"
            + table(pd.DataFrame(cutoff_audit))
            + "\n\nContrato: docs/OFFICIAL_CHALLENGE.md; fuente UBS fijada allí a commit 796d5805ec5f8a228a3ec0de36a2b4e6e1b1a1df. No hay merchant-family por evento.",
        ),
        (
            "5. Data processing",
            "JSONL: client_id, timestamp, amount, currency, description, direction, fee, mcc, type. Fechas UTC; texto lower/strip; MCC string; orden estable cliente/timestamp. El loader rechaza timestamps >= cutoff, duplicados exactos y solapamiento entre splits. Las etiquetas contienen cutoff_date y target. No hay conversión FX.\n\n"
            + table(pd.DataFrame(cutoff_audit)),
        ),
        ("6. Architecture", architecture),
        ("7. Features", feature_text),
        ("8. Models and heuristics", selection),
        (
            "9. Configuration selection",
            "El score comunicado es el máximo seleccionado sobre valid, no un test independiente. No se cambió la búsqueda histórica. La misma configuración se conserva en refit train+valid.\n\n"
            + runtime,
        ),
        ("10. Validation protocol", protocol),
        ("11. Reproduced metrics", main_result),
        ("12. F1 per class", class_table),
        (
            "13. Confusion matrix",
            cm_text + "\n\n![V1 confusion](figures/v1_baseline/v1_confusion.png)",
        ),
        ("14. Error profile", error_text),
        ("15. None analysis", none_text),
        ("16. Leakage audit", audit),
        ("17. Reproducibility", reproducibility),
        ("18. Tests", tests),
        ("19. Submission validation", submission_text),
        (
            "20. Strengths",
            "Control transparente, esquema estricto, ocho clases fijas, vocabulario aprendido en train, refit separado y ejecución reproducible. V1 constituye un fallback verificable.",
        ),
        (
            "21. Weaknesses",
            "Selección sobre valid; none infrapredicha; candidatos ML con target encoding in-sample; mezcla nominal de monedas; regularidad débil con pocos eventos; sin incertidumbre independiente. Ver auditoría para alcance exacto.",
        ),
        ("22. What V2 changes", comparison_arch),
        (
            "23. V1 vs V2",
            main_result
            + "\n\n"
            + table(pd.DataFrame([paired]))
            + "\n\n![Comparison](figures/v1_vs_v2/comparison.png)",
        ),
        (
            "24. Remaining uncertainty",
            "No hay etiquetas ocultas test ni evidencia del score del dashboard para estas nuevas submissions. El CSV mencionado no se adjuntó; se regeneró desde el commit V2. Las diferencias entre versiones no son una ablación causal. No se inventaron backtests con etiquetas históricas inexistentes.",
        ),
        ("25. Reproduction commands", commands),
        (
            "26. Final recommendation",
            "Conservar V1 como referencia. Preferir V2 para revisión del equipo por la mejora local reproducida; divulgar regresión de music y selección sobre valid. No interpretar la ganancia como promesa de leaderboard. No se envió submission ni se abrió PR.",
        ),
    ]
    save_report("v1_final_report.md", "V1 Final Report", sections)
    comparison = [
        ("Versions and artifacts compared", evidence_sources),
        ("Protocol comparability", protocol),
        (
            "Historical vs reproduced metrics",
            "Histórico V1: .2710243/.266; histórico V2: .391549456/.424. Valores frescos:\n\n"
            + main_result,
        ),
        ("Overall metrics", main_result),
        ("Per-class metrics", class_table),
        ("Confusion matrices", cm_text + "\n\n![V2 confusion](figures/v1_vs_v2/v2_confusion.png)"),
        ("Prediction changes", table(pd.DataFrame([paired]))),
        ("Errors corrected by V2", table(classes[["class", "corrected"]])),
        ("Errors introduced by V2", table(classes[["class", "introduced"]])),
        ("None transitions", none_text),
        ("Architecture and feature differences", comparison_arch),
        ("Evidence from existing ablations", findings),
        ("Runtime and complexity", runtime),
        ("Leakage and selection bias", audit),
        ("Submission validation", submission_text),
        (
            "What improved with evidence",
            main_result + "\n\n![Comparison](figures/v1_vs_v2/comparison.png)",
        ),
        (
            "What cannot be concluded",
            "No puede calcularse F1 sobre test sin etiquetas, ni atribuir causalmente todo el delta a una sola feature, ni interpretar el valid reutilizado como evaluación independiente.",
        ),
        (
            "Final recommendation",
            "V2 supera V1 localmente bajo protocolo comparable. Conservar ambas recetas congeladas y comunicar sus limitaciones.\n\n"
            + commands,
        ),
    ]
    save_report("v1_vs_v2_comparison.md", "V1 vs V2 Comparison", comparison)
    summary = f"""# V1 — Team Summary

V1 COMMIT: {V1}

V2 COMMIT: {V2}

V1 HISTORICAL MACRO-F1: 0.2710243

V1 REPRODUCED MACRO-F1: {m1['macro_f1']:.9f}

V1 REPRODUCED ACCURACY: {m1['accuracy']:.6f}

V2 COMPARABLE MACRO-F1: {m2['macro_f1']:.9f}

DELTA: {m2['macro_f1']-m1['macro_f1']:+.9f}

## V1 main components

- Mapa supervisado descripción/familia ajustado con train.
- Heurístico de recurrencia con 32 entradas familiares; bias none=-1, T=1.
- Selección sobre valid y refit del mapa sobre train+valid para test.

## Three strengths

1. Baseline reproducido desde un commit fijo.
2. Métrica oficial de ocho clases y alineación por ID.
3. Submission con contrato verificado y receta interpretable.

## Three weaknesses

1. Sesgo de selección por reutilizar valid.
2. none mal detectada; exceso de predicciones de algunas familias.
3. Candidatos ML originales con codificación influida por la propia etiqueta.

## What V2 improves

- Macro-F1 y accuracy suben bajo el mismo protocolo.
- {paired['only_v2_correct']} errores V1 corregidos frente a {paired['only_v1_correct']} aciertos V1 perdidos.
- none presenta la mayor mejora de F1.

## What V2 worsens

- Music pierde F1; la mejora global no beneficia a todas las clases.

## Main validation limitation

Valid se reutilizó en selección. No hay resultado oculto deducible del CSV test.

## Submission validation

Ambas reproducciones: 1.000 IDs únicos, columnas/clases/orden válidos.
El archivo V2 realmente enviado al dashboard no fue adjuntado en esta tarea.

## Recommendation

Conservar V1 como control y usar V2 para revisión del equipo por su mejora local.
Ver `v1_final_report.md`, `v1_vs_v2_comparison.md` y `v1_benchmark_checks.md`.
"""
    (REPORTS / "v1_final_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
