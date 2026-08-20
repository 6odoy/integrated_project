"""Metricas de la seccion 4.1 e intervalos bootstrap del Cuadro 3."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, f1_score

ESPEC_MIN = 0.80        # seccion 4.1: punto de decision con especificidad >= 0,80
N_BOOT = 1000
SEED = 42


def umbral_en_validacion(y_val, s_val, espec_min: float = ESPEC_MIN) -> float:
    """Umbral que maximiza sensibilidad sujeto a especificidad >= espec_min.

    Se calcula SOLO en validacion. Nunca se toca la prueba (seccion 4.1).
    """
    fpr, tpr, thr = roc_curve(y_val, s_val)
    ok = fpr <= (1.0 - espec_min)
    if not ok.any():
        return float(np.max(s_val)) + 1e-9
    return float(thr[ok][np.argmax(tpr[ok])])


def metricas(y, s, umbral: float) -> dict:
    y = np.asarray(y); s = np.asarray(s)
    pred = (s >= umbral).astype(int)
    pos, neg = y == 1, y == 0
    sens = pred[pos].mean() if pos.any() else np.nan
    espec = 1.0 - pred[neg].mean() if neg.any() else np.nan
    dos_clases = pos.any() and neg.any()
    return {
        "sens": sens,
        "espec": espec,
        "auc_roc": roc_auc_score(y, s) if dos_clases else np.nan,
        "auprc": average_precision_score(y, s) if dos_clases else np.nan,
        "f1": f1_score(y, pred, zero_division=0),
    }


def bootstrap_ic(y, s, grupos, umbral: float, n_boot: int = N_BOOT, seed: int = SEED):
    """IC al 95 % con 1.000 remuestreos, a nivel de PACIENTE.

    Se remuestrean pacientes, no imagenes: las imagenes del mismo paciente estan
    correlacionadas y remuestrearlas sueltas estrecharia el intervalo de mas.
    """
    y = np.asarray(y); s = np.asarray(s); grupos = np.asarray(grupos)
    pacientes = np.unique(grupos)
    idx_por_pac = {p: np.flatnonzero(grupos == p) for p in pacientes}
    rng = np.random.default_rng(seed)

    acum, descartados = [], 0
    for _ in range(n_boot):
        muestra = rng.choice(pacientes, size=len(pacientes), replace=True)
        idx = np.concatenate([idx_por_pac[p] for p in muestra])
        yb = y[idx]
        if len(np.unique(yb)) < 2:      # replica degenerada
            descartados += 1
            continue
        acum.append(metricas(yb, s[idx], umbral))

    if not acum:
        return {}, n_boot
    ic = {}
    for k in acum[0]:
        v = np.array([a[k] for a in acum], dtype=float)
        v = v[~np.isnan(v)]
        ic[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)
    return ic, descartados
