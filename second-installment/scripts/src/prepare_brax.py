"""Prepara BRAX para transferencia local y una evaluación externa honesta.

La etiqueta positiva es ``Pneumonia=1``. La negativa es deliberadamente más
estricta: ``No Finding=1`` y ``Pneumonia=0``. Se excluyen etiquetas inciertas,
vistas no frontales y estudios que no cumplen una de esas dos definiciones.
Los pacientes se asignan una única vez, antes de entrenar, a adapt_train,
adapt_val o external_test; por tanto el último split no debe usarse para tomar
decisiones de modelamiento.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="CSV de manifiesto ya particionado")
    ap.add_argument("--patient-col", default="patient_id")
    ap.add_argument("--study-col", default="study_id")
    ap.add_argument("--path-col", default="image_path")
    ap.add_argument("--view-col", default="view")
    ap.add_argument("--pneumonia-col", default="Pneumonia")
    ap.add_argument("--no-finding-col", default="No Finding")
    ap.add_argument("--seed", type=int, default=42, help="Semilla de la asignación por paciente")
    ap.add_argument("--adapt-train-fraction", type=float, default=.70)
    ap.add_argument("--adapt-val-fraction", type=float, default=.15)
    return ap.parse_args()


def validate_columns(frame: pd.DataFrame, args: argparse.Namespace) -> None:
    columns = {args.patient_col, args.study_col, args.path_col, args.view_col,
               args.pneumonia_col, args.no_finding_col}
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"El metadata no contiene las columnas requeridas: {sorted(missing)}")


def assign_patients(frame: pd.DataFrame, seed: int, train_fraction: float,
                    val_fraction: float) -> pd.DataFrame:
    """Estratifica por etiqueta máxima del paciente y nunca divide un paciente."""
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 or train_fraction + val_fraction >= 1:
        raise ValueError("Las proporciones deben ser positivas y sumar menos de 1")
    patients = frame.groupby("patient_id", as_index=False).label.max()
    rng, assignment = np.random.default_rng(seed), {}
    for label in (0, 1):
        ids = patients.loc[patients.label == label, "patient_id"].to_numpy(copy=True)
        if len(ids) < 3:
            raise ValueError(f"Hay solo {len(ids)} pacientes de clase {label}; se requieren al menos 3")
        rng.shuffle(ids)
        train_end = min(max(round(train_fraction * len(ids)), 1), len(ids) - 2)
        val_end = min(max(train_end + round(val_fraction * len(ids)), train_end + 1), len(ids) - 1)
        assignment.update({patient: "adapt_train" for patient in ids[:train_end]})
        assignment.update({patient: "adapt_val" for patient in ids[train_end:val_end]})
        assignment.update({patient: "external_test" for patient in ids[val_end:]})
    result = frame.copy()
    result["split"] = result.patient_id.map(assignment)
    if result.split.isna().any():
        raise RuntimeError("Hay imágenes sin asignación de split")
    patient_splits = result.groupby("patient_id").split.nunique()
    if (patient_splits != 1).any():
        raise RuntimeError("LEAKAGE: un paciente aparece en más de un split")
    if set(result.split.unique()) != {"adapt_train", "adapt_val", "external_test"}:
        raise RuntimeError("La partición no contiene los tres splits requeridos")
    return result


def split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = frame.groupby("split", sort=False).agg(
        images=("label", "size"), patients=("patient_id", "nunique"),
        studies=("study_id", "nunique"), positives=("label", "sum"), prevalence=("label", "mean"),
    )
    return summary.reindex(["adapt_train", "adapt_val", "external_test"])


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.metadata)
    validate_columns(raw, args)
    log = [{"stage": "raw", "images": len(raw), "patients": raw[args.patient_col].nunique()}]
    front = raw[raw[args.view_col].astype(str).str.upper().isin({"PA", "AP"})].copy()
    log.append({"stage": "frontal_PA_AP", "images": len(front), "patients": front[args.patient_col].nunique()})
    pneumonia, no_finding = front[args.pneumonia_col], front[args.no_finding_col]
    eligible = front[(pneumonia.isin([0, 1])) & (no_finding.isin([0, 1])) &
                     (((pneumonia == 1) & (no_finding == 0)) |
                      ((pneumonia == 0) & (no_finding == 1)))].copy()
    eligible["label"] = eligible[args.pneumonia_col].astype(int)
    manifest = eligible.rename(columns={args.patient_col: "patient_id", args.study_col: "study_id",
                                        args.path_col: "image_path", args.view_col: "view"})[
        ["patient_id", "study_id", "image_path", "view", "label"]]
    log.append({"stage": "binary_eligible", "images": len(manifest),
                "patients": manifest.patient_id.nunique(), "prevalence": manifest.label.mean()})
    partition = assign_patients(manifest, args.seed, args.adapt_train_fraction, args.adapt_val_fraction)
    summary = split_summary(partition)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partition.to_csv(args.out, index=False)
    pd.DataFrame(log).to_csv(args.out.with_name("brax_filter_flow.csv"), index=False)
    summary.to_csv(args.out.with_name("brax_partition_summary.csv"))
    protocol = {
        "seed": args.seed, "patient_unit": "patient_id", "stratification_label": "maximum image label per patient",
        "fractions": {"adapt_train": args.adapt_train_fraction, "adapt_val": args.adapt_val_fraction,
                      "external_test": 1 - args.adapt_train_fraction - args.adapt_val_fraction},
        "label_definition": {"positive": "Pneumonia=1 and No Finding=0",
                               "negative": "Pneumonia=0 and No Finding=1",
                               "excluded": "uncertain/missing labels, non-frontal views, and all other label combinations"},
        "selection_rule": "Split assignment is generated once from seed before training; external_test is never used for training, early stopping, threshold selection, or model selection.",
        "leakage_check": "passed",
    }
    args.out.with_name("brax_partition_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf8")
    print(pd.DataFrame(log).to_string(index=False))
    print("\nPartición por paciente (sin fuga):")
    print(summary.to_string())


if __name__ == "__main__":
    main()
