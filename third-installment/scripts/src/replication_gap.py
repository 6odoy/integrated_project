"""Explica con evidencia la brecha de especificidad frente a Rahman et al. (0,970).

Dos hipótesis, cada una con su prueba:

1. Punto de operación. El protocolo elige el umbral de máxima sensibilidad con
   especificidad >= 0,80; el artículo informa el punto por defecto. Se mide,
   sobre las predicciones ya guardadas, la especificidad en 0,5 y la
   sensibilidad que tendría el modelo a la especificidad del artículo. Es una
   lectura descriptiva del test, no una selección: el umbral oficial no cambia.
2. Fuga por paciente. El artículo parte por imagen. ``--make-image-split``
   genera una partición 70/15/15 por imagen con la misma semilla; entrenada con
   train.py, su diferencia con la partición por paciente estima cuánto
   rendimiento aporta que un mismo paciente aparezca en train y test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

RAHMAN = {"sensitivity": .990, "specificity": .970, "f1": .981}


def make_image_split(manifest: Path, out: Path, seed: int) -> None:
    frame = pd.read_csv(manifest)
    frame = frame[frame.label.isin([0, 1])].copy()
    rng, split = np.random.default_rng(seed), pd.Series(index=frame.index, dtype=object)
    for label in (0, 1):
        index = frame.index[frame.label == label].to_numpy(copy=True)
        rng.shuffle(index)
        a, b = round(.70 * len(index)), round(.85 * len(index))
        split[index[:a]], split[index[a:b]], split[index[b:]] = "train", "val", "test"
    frame["split"] = split
    shared = frame.groupby("patient_id").split.nunique()
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    test_patients = set(frame.patient_id[frame.split == "test"])
    train_patients = set(frame.patient_id[frame.split == "train"])
    print(f"Partición por imagen: {int((shared > 1).sum())} pacientes en más de un split; "
          f"{len(test_patients & train_patients)} de {len(test_patients)} pacientes de test también están en train")


def operating_points(y, s, threshold):
    fpr, tpr, thresholds = roc_curve(y, s)
    at_spec = tpr[fpr <= 1 - RAHMAN["specificity"]].max()
    at_sens = 1 - fpr[tpr >= RAHMAN["sensitivity"]].min()
    p05 = s >= .5
    return {"auc_roc": float(roc_auc_score(y, s)),
            "protocol_threshold": float(threshold),
            "sensitivity_protocol": float((s[y == 1] >= threshold).mean()),
            "specificity_protocol": float((s[y == 0] < threshold).mean()),
            "sensitivity_at_0.5": float(p05[y == 1].mean()), "specificity_at_0.5": float(1 - p05[y == 0].mean()),
            "sensitivity_at_rahman_specificity": float(at_spec),
            "specificity_at_rahman_sensitivity": float(at_sens)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kermany-results", type=Path, help="second-installment/scripts/results/kermany")
    ap.add_argument("--image-split-results", type=Path, help="Corridas entrenadas con la partición por imagen")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--make-image-split", type=Path, help="Manifiesto Kermany; genera la partición por imagen")
    ap.add_argument("--image-split-out", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.make_image_split:
        make_image_split(args.make_image_split, args.image_split_out, args.seed)
        return
    rows = []
    for split_name, root in (("patient", args.kermany_results), ("image", args.image_split_results)):
        if root is None:
            continue
        for path in sorted(root.glob("*/seed*/predictions_kermany_test.csv")):
            mode = path.parent.parent.name
            if mode not in ("frozen", "finetune_partial"):
                continue
            metrics = json.loads((path.parent / "metrics.json").read_text())
            p = pd.read_csv(path)
            rows.append({"partition": split_name, "mode": mode, "seed": metrics["seed"],
                         **operating_points(p.label.to_numpy(), p.score.to_numpy(), metrics["threshold"])})
    frame = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "replication_gap_by_run.csv", index=False)
    summary = frame.groupby(["partition", "mode"]).agg(["mean", lambda v: v.std(ddof=0)]).rename(columns={"<lambda_0>": "std"}).drop(columns="seed")
    summary.to_csv(args.out / "replication_gap_summary.csv")
    write_table(frame[frame["mode"] == "finetune_partial"], args.out / "table_replication.tex")
    print(frame.groupby(["partition", "mode"]).mean(numeric_only=True).round(3).T.to_string())


def write_table(frame: pd.DataFrame, path: Path) -> None:
    """Fine-tuning parcial por partición: media y DE de las semillas, coma decimal."""
    def cell(values):
        text = f"{values.mean():.3f}" + (f" $\\pm$ {values.std(ddof=0):.3f}" if len(values) > 1 else "")
        return text.replace(".", "{,}")
    rows = [r"Rahman et al. (DenseNet201, partición por imagen) & -- & 0{,}990 / 0{,}970 & -- & -- \\"]
    names = {"patient": "Esta réplica, partición por paciente", "image": "Esta réplica, partición por imagen"}
    for key in ("patient", "image"):
        part = frame[frame.partition == key]
        if part.empty:
            continue
        rows.append(f"{names[key]} ({len(part)} sem.) & {cell(part.auc_roc)} & "
                    f"{cell(part.sensitivity_protocol)} / {cell(part.specificity_protocol)} & "
                    f"{cell(part['sensitivity_at_0.5'])} / {cell(part['specificity_at_0.5'])} & "
                    f"{cell(part.specificity_at_rahman_sensitivity)} \\\\")
    path.write_text("\\begin{tabular}{lcccc}\n\\toprule\n"
                    "Fuente & AUC-ROC & Sens. / Espec. (protocolo) & Sens. / Espec. (umbral 0,5) & Espec. a sens. 0,99 \\\\\n"
                    "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
