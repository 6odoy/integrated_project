"""Utilidades del traslado a NIH: carga de la partición, loaders y evaluación externa."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PHASE2_SRC = Path(__file__).resolve().parents[3] / "second-installment" / "scripts" / "src"
sys.path.insert(0, str(PHASE2_SRC))
from common import metrics, patient_bootstrap  # noqa: E402
from train import XrayDataset, predict, transforms_for  # noqa: E402

SPLITS = ("adapt_train", "adapt_val", "external_test")
METADATA = ["view", "age", "sex", "finding_labels", "pneumonia_only", "has_bbox", "audit_reserved"]


def load_partition(path: Path) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(path)
    if set(frame.split) != set(SPLITS):
        raise ValueError("Se requiere partition_nih.csv generado por prepare_nih.py")
    if (frame.groupby("patient_id").split.nunique() > 1).any():
        raise RuntimeError("LEAKAGE: un paciente pertenece a más de un split")
    return {name: frame[frame.split == name].reset_index(drop=True) for name in SPLITS}


def subsample_patients(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Submuestra estratificada por paciente para la curva de datos escasos."""
    if fraction >= 1:
        return frame
    patients = frame.groupby("patient_id", as_index=False).label.max()
    rng, keep = np.random.default_rng(seed), []
    for label in (0, 1):
        ids = patients.loc[patients.label == label, "patient_id"].to_numpy(copy=True)
        rng.shuffle(ids)
        keep.extend(ids[:max(1, round(fraction * len(ids)))])
    return frame[frame.patient_id.isin(keep)].reset_index(drop=True)


def make_loader(frame, images_root, image_size, batch_size, train=False, seed=0):
    generator = torch.Generator().manual_seed(seed) if train else None
    dataset = XrayDataset(frame, images_root, transforms_for(train), cache_resize=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0, generator=generator)


def evaluate_split(model, loader, frame, threshold, device, n_boot, seed):
    """Métricas, IC bootstrap por paciente y predicciones con metadatos para la auditoría."""
    y, scores, patients, studies = predict(model, loader, device)
    if not (studies == frame.study_id.astype(str).to_numpy()).all():
        raise RuntimeError("El orden de las predicciones no coincide con la partición")
    result = metrics(y, scores, threshold)
    result["bootstrap_95"] = patient_bootstrap(y, scores, patients, threshold, n_boot, seed)
    random_part = ~frame.audit_reserved.to_numpy()
    result["random_subset_without_bbox_reserve"] = metrics(y[random_part], scores[random_part], threshold)
    predictions = frame[["patient_id", "study_id", "label"] + METADATA].assign(
        score=scores, prediction=(scores >= threshold).astype(int))
    return result, predictions
