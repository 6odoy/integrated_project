"""Linea base Random Forest sobre la particion exacta de Fase 2.

Guarda predicciones por imagen y el umbral escogido solo en validacion para que
``compare_paired.py`` pueda contrastarla honestamente con una CNN.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from common import metrics, patient_bootstrap, select_threshold, verify_no_leakage, write_json

N_BINS, IMAGE_SIZE = 64, 128


def intensity_histograms(frame: pd.DataFrame, image_root: Path) -> np.ndarray:
    """Las mismas caracteristicas de la linea base de Fase 1 (sin CNN)."""
    features = np.empty((len(frame), N_BINS), dtype=np.float32)
    for row_index, relative_path in enumerate(frame.image_path):
        image = Image.open(image_root / relative_path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
        histogram, _ = np.histogram(np.asarray(image, dtype=np.uint8), bins=N_BINS, range=(0, 256))
        features[row_index] = histogram / histogram.sum()
    return features


def select_random_forest(x_train, y_train, x_val, y_val, seed):
    """Misma rejilla declarada en Fase 1; se selecciona exclusivamente por AUC-val."""
    best = (-np.inf, None, None)
    for n_estimators in (300, 600):
        for max_depth in (None, 10, 20):
            for min_samples_leaf in (1, 5):
                model = RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf, class_weight="balanced",
                    n_jobs=-1, random_state=seed,
                ).fit(x_train, y_train)
                auc = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])
                if auc > best[0]:
                    best = (auc, model, {"n_estimators": n_estimators,
                                         "max_depth": max_depth,
                                         "min_samples_leaf": min_samples_leaf})
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", required=True, type=Path)
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    frame = pd.read_csv(args.partition)
    expected = {"train", "val", "test"}
    if set(frame.split.unique()) != expected:
        raise ValueError(f"La particion debe contener exactamente {sorted(expected)}")
    verify_no_leakage(frame)
    splits = {split: frame[frame.split == split].reset_index(drop=True) for split in expected}
    features = {split: intensity_histograms(splits[split], args.images_root) for split in expected}
    labels = {split: splits[split].label.to_numpy(dtype=int) for split in expected}
    val_auc, model, hyperparameters = select_random_forest(
        features["train"], labels["train"], features["val"], labels["val"], args.seed)
    val_scores = model.predict_proba(features["val"])[:, 1]
    threshold = select_threshold(labels["val"], val_scores)
    test_scores = model.predict_proba(features["test"])[:, 1]
    test = splits["test"]
    result = metrics(labels["test"], test_scores, threshold)
    result.update({
        "model": "Random Forest sobre histograma de intensidad (Fase 1)",
        "seed": args.seed,
        "hyperparameters_selected_on_validation": hyperparameters,
        "best_validation_auc": float(val_auc),
        "threshold": float(threshold),
        "threshold_from": "Kermany validation only; specificity_minimum=0.80",
        "patients": int(test.patient_id.nunique()), "images": int(len(test)),
        "prevalence": float(labels["test"].mean()),
        "partition": str(args.partition),
    })
    result["bootstrap_95"] = patient_bootstrap(
        labels["test"], test_scores, test.patient_id.to_numpy(), threshold,
        args.n_boot, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"patient_id": test.patient_id, "study_id": test.study_id,
                  "label": labels["test"], "score": test_scores,
                  "prediction": (test_scores >= threshold).astype(int)}).to_csv(
                      args.out / "predictions_kermany_test.csv", index=False)
    write_json(args.out / "metrics.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
