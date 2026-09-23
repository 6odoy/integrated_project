"""Utilidades compartidas de Fase 2: partición, métricas e intervalos."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

ESPEC_MIN = 0.80

def split_by_patient(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Partición 70/15/15 por paciente, estratificada con la etiqueta máxima."""
    required = {"patient_id", "study_id", "image_path", "view", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    df = df[df.view.str.upper().isin({"PA", "AP"}) & df.label.isin([0, 1])].copy()
    patients = df.groupby("patient_id", as_index=False).label.max()
    rng = np.random.default_rng(seed)
    assignment = {}
    for label in (0, 1):
        ids = patients.loc[patients.label == label, "patient_id"].to_numpy()
        rng.shuffle(ids)
        a, b = round(.70 * len(ids)), round(.85 * len(ids))
        assignment.update({x: "train" for x in ids[:a]})
        assignment.update({x: "val" for x in ids[a:b]})
        assignment.update({x: "test" for x in ids[b:]})
    df["split"] = df.patient_id.map(assignment)
    verify_no_leakage(df)
    return df

def verify_no_leakage(df: pd.DataFrame) -> None:
    counts = df.groupby("patient_id").split.nunique()
    leaked = counts[counts > 1]
    if not leaked.empty:
        raise RuntimeError(f"LEAKAGE: {len(leaked)} pacientes pertenecen a más de un split")

def select_threshold(y, scores, specificity_min=ESPEC_MIN):
    fpr, tpr, thresholds = roc_curve(y, scores)
    valid = fpr <= 1 - specificity_min
    if not valid.any():
        return float(np.max(scores) + 1e-8)
    return float(thresholds[valid][np.argmax(tpr[valid])])

def metrics(y, scores, threshold):
    y, scores = np.asarray(y), np.asarray(scores)
    prediction = (scores >= threshold).astype(int)
    pos, neg = y == 1, y == 0
    specificity = float(1 - prediction[neg].mean()) if neg.any() else np.nan
    sensitivity = float(prediction[pos].mean()) if pos.any() else np.nan
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        # La restriccion se impone al elegir el umbral en validacion. Esta
        # variante evita presentar sensibilidad de un punto que, al evaluarse,
        # no alcanza la especificidad minima (por ejemplo el trivial positivo).
        "sensitivity_at_specificity_minimum": sensitivity if np.isnan(specificity) or specificity >= ESPEC_MIN else np.nan,
        "specificity_constraint_satisfied": bool(np.isnan(specificity) or specificity >= ESPEC_MIN),
        "auc_roc": float(roc_auc_score(y, scores)) if pos.any() and neg.any() else np.nan,
        "auprc": float(average_precision_score(y, scores)) if pos.any() and neg.any() else np.nan,
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "tp": int(((prediction == 1) & pos).sum()), "fn": int(((prediction == 0) & pos).sum()),
        "tn": int(((prediction == 0) & neg).sum()), "fp": int(((prediction == 1) & neg).sum()),
    }

def patient_bootstrap(y, scores, patient_ids, threshold, n_boot=1000, seed=42):
    y, scores, patient_ids = np.asarray(y), np.asarray(scores), np.asarray(patient_ids)
    patient_ids_unique = np.unique(patient_ids)
    index = {p: np.flatnonzero(patient_ids == p) for p in patient_ids_unique}
    rng, rows = np.random.default_rng(seed), []
    for _ in range(n_boot):
        sample = rng.choice(patient_ids_unique, len(patient_ids_unique), replace=True)
        ix = np.concatenate([index[p] for p in sample])
        if len(np.unique(y[ix])) == 2:
            rows.append(metrics(y[ix], scores[ix], threshold))
    return {k: [float(np.percentile([r[k] for r in rows], q)) for q in (2.5, 97.5)]
            for k in ("sensitivity", "specificity", "sensitivity_at_specificity_minimum", "auc_roc", "auprc", "f1")
            if any(not np.isnan(r[k]) for r in rows)} if rows else {}

def paired_patient_bootstrap_difference(y, scores_a, threshold_a, scores_b,
                                        threshold_b, patient_ids, n_boot=1000,
                                        seed=42):
    """IC percentil para A - B remuestreando los mismos pacientes en ambos.

    Las predicciones deben corresponder a exactamente las mismas imagenes. Al
    compartir cada remuestra se conserva la correlacion entre modelos y el IC
    estima la *diferencia* de desempeno, no el solapamiento de dos IC separados.
    """
    y = np.asarray(y)
    scores_a, scores_b = np.asarray(scores_a), np.asarray(scores_b)
    patient_ids = np.asarray(patient_ids)
    if not (len(y) == len(scores_a) == len(scores_b) == len(patient_ids)):
        raise ValueError("y, scores y patient_ids deben tener la misma longitud")
    patients = np.unique(patient_ids)
    if not len(patients):
        raise ValueError("No hay pacientes para bootstrap pareado")
    indices = {p: np.flatnonzero(patient_ids == p) for p in patients}
    rng = np.random.default_rng(seed)
    differences, discarded = [], 0
    metric_names = ("sensitivity", "specificity", "auc_roc", "auprc", "f1")
    for _ in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        ix = np.concatenate([indices[p] for p in sampled])
        if np.unique(y[ix]).size < 2:
            discarded += 1
            continue
        ma = metrics(y[ix], scores_a[ix], threshold_a)
        mb = metrics(y[ix], scores_b[ix], threshold_b)
        differences.append({name: ma[name] - mb[name] for name in metric_names})
    if not differences:
        return {"n_boot_requested": n_boot, "n_boot_valid": 0,
                "n_boot_discarded_degenerate": discarded, "metrics": {}}
    return {
        "n_boot_requested": n_boot,
        "n_boot_valid": len(differences),
        "n_boot_discarded_degenerate": discarded,
        "metrics": {
            name: {
                "estimate": float(metrics(y, scores_a, threshold_a)[name] - metrics(y, scores_b, threshold_b)[name]),
                "ci_95": [float(np.percentile([row[name] for row in differences], q)) for q in (2.5, 97.5)],
            }
            for name in metric_names
        },
    }

def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf8")
