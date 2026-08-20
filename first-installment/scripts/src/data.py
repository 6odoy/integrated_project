"""Carga, filtrado, particion por paciente y extraccion de caracteristicas.

Contrato del manifiesto (CSV) - una fila por imagen:
    patient_id, study_id, image_path, view, label
      view  : PA | AP | LATERAL | ...   (solo se conservan PA y AP)
      label : 1 = Neumonia, 0 = Sin hallazgos, -1 = incertidumbre
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from PIL import Image

SEED = 42
IMG_SIZE = 128          # linea base, seccion 4.2
N_BINS = 64             # histograma de intensidad
FRONTAL = {"PA", "AP"}


def load_manifest(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (df_modelable, df_incertidumbre)."""
    df = pd.read_csv(path)
    faltan = {"patient_id", "image_path", "view", "label"} - set(df.columns)
    if faltan:
        raise ValueError(f"El manifiesto no tiene las columnas: {sorted(faltan)}")

    n0 = len(df)
    df = df[df["view"].str.upper().isin(FRONTAL)].copy()
    print(f"  vistas frontales (PA/AP): {len(df)}/{n0}")

    incert = df[df["label"] == -1].copy()          # reservado para Fase 3
    df = df[df["label"].isin([0, 1])].copy()
    print(f"  modelables: {len(df)}  |  incertidumbre apartada: {len(incert)}")
    return df, incert


def split_por_paciente(df: pd.DataFrame, seed: int = SEED):
    """70/15/15 por PACIENTE (no por imagen), estratificado por etiqueta de paciente.

    Un paciente es positivo si tiene al menos un estudio positivo. Esto es lo que
    impide el data leakage descrito en la seccion 3.3.
    """
    pac = df.groupby("patient_id")["label"].max().reset_index()
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for etiqueta in (0, 1):
        ids = pac.loc[pac["label"] == etiqueta, "patient_id"].to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        i1, i2 = int(round(0.70 * n)), int(round(0.85 * n))
        tr += list(ids[:i1]); va += list(ids[i1:i2]); te += list(ids[i2:])

    part = {"train": set(tr), "val": set(va), "test": set(te)}
    df = df.copy()
    df["split"] = df["patient_id"].map(
        lambda p: next(k for k, v in part.items() if p in v))

    # Verificacion dura: ningun paciente puede cruzar el limite.
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        cruce = part[a] & part[b]
        assert not cruce, f"LEAKAGE: {len(cruce)} pacientes en {a} y {b}"
    return df


def caracteristicas(df: pd.DataFrame, images_root: str = "") -> np.ndarray:
    """Histograma de intensidad de N_BINS sobre la imagen en escala de grises 128x128."""
    import os
    X = np.zeros((len(df), N_BINS), dtype=np.float32)
    for i, rel in enumerate(df["image_path"].to_numpy()):
        ruta = os.path.join(images_root, rel) if images_root else rel
        img = Image.open(ruta).convert("L").resize((IMG_SIZE, IMG_SIZE))
        h, _ = np.histogram(np.asarray(img, dtype=np.uint8),
                            bins=N_BINS, range=(0, 256))
        X[i] = h / h.sum()
    return X
