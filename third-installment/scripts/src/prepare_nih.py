"""Selecciona el subconjunto NIH ChestX-ray14 y lo particiona por paciente.

Reemplaza a BRAX como conjunto de traslado y conserva su protocolo:

* Población adulta (edad >= 18) y vistas frontales PA/AP: el traslado es
  pediátrico (Kermany) -> adulto.
* Positivo: ``Pneumonia`` aparece en ``Finding Labels``, con o sin otros
  hallazgos. Negativo: ``Finding Labels == "No Finding"``. Todo lo demás se
  excluye.
* Diseño caso-control por paciente. Un paciente con neumonía en cualquier
  imagen aporta solo sus imágenes con neumonía; un control nunca tiene
  neumonía en ninguna imagen del conjunto completo y aporta una única imagen
  ``No Finding`` elegida al azar. Se muestrean ``--controls-per-case``
  imágenes de control por imagen positiva.
* Los pacientes caso se reparten 70/15/15 con semilla fija. Los pacientes con
  caja de neumonía del radiólogo (BBox_List_2017.csv) se reservan en
  ``external_test`` *además* del 15 % aleatorio: se usan para auditar Grad-CAM
  y nunca entran al ajuste ni a la selección. Los controles se reparten después
  para que la prevalencia por imagen sea idéntica en los tres splits.

Todo se decide desde los metadatos, antes de descargar una sola imagen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("adapt_train", "adapt_val", "external_test")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True, type=Path, help="Data_Entry_2017_v2020.csv")
    ap.add_argument("--bbox", required=True, type=Path, help="BBox_List_2017.csv")
    ap.add_argument("--out", required=True, type=Path, help="Directorio de salida")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-age", type=int, default=18)
    ap.add_argument("--controls-per-case", type=float, default=3.0)
    ap.add_argument("--train-fraction", type=float, default=.70)
    ap.add_argument("--val-fraction", type=float, default=.15)
    return ap.parse_args()


def load_metadata(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    # La versión v2020 de NIH Box usa "Patient Gender"; algunos espejos, "Patient Sex".
    raw = raw.rename(columns={"Patient Gender": "Patient Sex"})
    required = {"Image Index", "Finding Labels", "Patient ID", "Patient Age", "Patient Sex", "View Position"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el metadata: {sorted(missing)}")
    return raw


def select_case_control(raw: pd.DataFrame, min_age: int, controls_per_case: float, seed: int):
    log = [{"stage": "raw", "images": len(raw), "patients": raw["Patient ID"].nunique()}]
    has_pneumonia = raw["Finding Labels"].str.split("|").apply(lambda labels: "Pneumonia" in labels)
    ever_pneumonia = set(raw.loc[has_pneumonia, "Patient ID"])
    eligible = raw[raw["View Position"].isin(["PA", "AP"]) & (raw["Patient Age"] >= min_age)]
    log.append({"stage": f"frontal_PA_AP_age>={min_age}", "images": len(eligible),
                "patients": eligible["Patient ID"].nunique()})
    cases = eligible[has_pneumonia.loc[eligible.index]].assign(label=1)
    log.append({"stage": "cases_pneumonia", "images": len(cases), "patients": cases["Patient ID"].nunique()})
    pool = eligible[eligible["Finding Labels"].eq("No Finding") & ~eligible["Patient ID"].isin(ever_pneumonia)]
    log.append({"stage": "control_pool_no_finding_never_pneumonia", "images": len(pool),
                "patients": pool["Patient ID"].nunique()})
    rng = np.random.default_rng(seed)
    # Una imagen por paciente control, elegida al azar con la semilla del protocolo.
    one_per_patient = pool.loc[pool.groupby("Patient ID").apply(
        lambda g: g.index[rng.integers(len(g))], include_groups=False).to_numpy()]
    n_controls = int(round(controls_per_case * len(cases)))
    if n_controls > len(one_per_patient):
        raise ValueError(f"Se piden {n_controls} controles, pero solo hay {len(one_per_patient)} pacientes")
    controls = one_per_patient.iloc[rng.permutation(len(one_per_patient))[:n_controls]].assign(label=0)
    log.append({"stage": "controls_sampled_one_image_per_patient", "images": len(controls),
                "patients": controls["Patient ID"].nunique()})
    selected = pd.concat([cases, controls]).sort_values("Image Index")
    log.append({"stage": "selected", "images": len(selected), "patients": selected["Patient ID"].nunique(),
                "prevalence": float(selected.label.mean())})
    return selected, log


def assign_patients(frame: pd.DataFrame, reserved: set, seed: int, train_fraction: float,
                    val_fraction: float) -> pd.DataFrame:
    """Reparte pacientes caso 70/15/15 y luego controles para igualar la prevalencia.

    Los casos reservados van siempre a external_test. Como cada control aporta
    una sola imagen, cada split recibe tantos controles como imágenes caso por
    la razón global de controles por caso; así la prevalencia por imagen es la
    misma en los tres splits y el umbral no hereda un cambio de prevalencia.
    """
    patients = frame.groupby("patient_id", as_index=False).label.max()
    rng, assignment = np.random.default_rng(seed), {}
    cases = patients.loc[(patients.label == 1) & ~patients.patient_id.isin(reserved), "patient_id"].to_numpy(copy=True)
    rng.shuffle(cases)
    train_end = round(train_fraction * len(cases))
    val_end = train_end + round(val_fraction * len(cases))
    assignment.update({p: "adapt_train" for p in cases[:train_end]})
    assignment.update({p: "adapt_val" for p in cases[train_end:val_end]})
    assignment.update({p: "external_test" for p in cases[val_end:]})
    assignment.update({p: "external_test" for p in reserved})
    positives = frame[frame.label == 1]
    case_images = positives.patient_id.map(assignment).value_counts()
    controls = patients.loc[patients.label == 0, "patient_id"].to_numpy(copy=True)
    rng.shuffle(controls)
    ratio = len(controls) / len(positives)
    n_val = round(ratio * case_images["adapt_val"])
    n_test = round(ratio * case_images["external_test"])
    assignment.update({p: "adapt_val" for p in controls[:n_val]})
    assignment.update({p: "external_test" for p in controls[n_val:n_val + n_test]})
    assignment.update({p: "adapt_train" for p in controls[n_val + n_test:]})
    result = frame.assign(split=frame.patient_id.map(assignment))
    if result.split.isna().any():
        raise RuntimeError("Hay imágenes sin split")
    if (result.groupby("patient_id").split.nunique() != 1).any():
        raise RuntimeError("LEAKAGE: un paciente aparece en más de un split")
    if set(result.split) != set(SPLITS):
        raise RuntimeError("La partición no contiene los tres splits")
    return result


def main() -> None:
    args = parse_args()
    raw = load_metadata(args.metadata)
    selected, log = select_case_control(raw, args.min_age, args.controls_per_case, args.seed)
    bbox = pd.read_csv(args.bbox)
    bbox_images = set(bbox.loc[bbox["Finding Label"] == "Pneumonia", "Image Index"])
    manifest = pd.DataFrame({
        "patient_id": "nih_" + selected["Patient ID"].astype(str),
        "study_id": selected["Image Index"].str.replace(".png", "", regex=False),
        "image_path": selected["Image Index"],
        "view": selected["View Position"],
        "label": selected.label.astype(int),
        "age": selected["Patient Age"].astype(int),
        "sex": selected["Patient Sex"],
        "finding_labels": selected["Finding Labels"],
        "pneumonia_only": selected["Finding Labels"].eq("Pneumonia"),
        "has_bbox": selected["Image Index"].isin(bbox_images),
    })
    reserved = set(manifest.loc[manifest.has_bbox, "patient_id"])
    partition = assign_patients(manifest, reserved, args.seed, args.train_fraction, args.val_fraction)
    partition["audit_reserved"] = partition.patient_id.isin(reserved)
    args.out.mkdir(parents=True, exist_ok=True)
    partition.to_csv(args.out / "partition_nih.csv", index=False)
    pd.DataFrame(log).to_csv(args.out / "nih_filter_flow.csv", index=False)
    summary = partition.groupby("split").agg(
        images=("label", "size"), patients=("patient_id", "nunique"), positives=("label", "sum"),
        prevalence=("label", "mean"), ap_fraction=("view", lambda v: float((v == "AP").mean())),
        bbox_images=("has_bbox", "sum")).reindex(list(SPLITS))
    summary.to_csv(args.out / "partition_nih_summary.csv")
    protocol = {
        "source_metadata": "Data_Entry_2017_v2020.csv (NIH Clinical Center, nihcc.app.box.com/v/ChestXray-NIHCC)",
        "seed": args.seed, "min_age": args.min_age, "views": ["PA", "AP"],
        "label_definition": {"positive": "'Pneumonia' in Finding Labels (co-findings allowed)",
                             "negative": "Finding Labels == 'No Finding' and patient never labeled Pneumonia",
                             "excluded": "pediatric, non-frontal and every other label combination"},
        "design": f"case-control; {args.controls_per_case} control images per case image; one image per control patient",
        "case_patient_fractions": {"adapt_train": args.train_fraction, "adapt_val": args.val_fraction,
                                   "external_test": round(1 - args.train_fraction - args.val_fraction, 6)},
        "control_allocation": "controls per split = global control/case ratio x case images in that split",
        "reserved_for_audit": f"{len(reserved)} patients with radiologist Pneumonia bounding boxes forced into external_test",
        "selection_rule": "Decided from metadata before any image is downloaded or any model is trained; "
                          "external_test never informs training, early stopping, threshold or model selection.",
        "leakage_check": "passed",
    }
    (args.out / "nih_partition_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf8")
    print(pd.DataFrame(log).to_string(index=False))
    print("\nPartición por paciente (sin fuga):")
    print(summary.to_string())


if __name__ == "__main__":
    main()
