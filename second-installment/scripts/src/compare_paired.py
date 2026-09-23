"""Compara dos modelos sobre las mismas imágenes mediante bootstrap pareado."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import metrics, paired_patient_bootstrap_difference, write_json

ap = argparse.ArgumentParser()
ap.add_argument("--model-a", required=True, type=Path, help="CSV con patient_id, study_id, label, score")
ap.add_argument("--threshold-a", type=float, help="Umbral explícito de A")
ap.add_argument("--metrics-a", type=Path, help="metrics.json del que leer el umbral de A")
ap.add_argument("--model-b", required=True, type=Path)
ap.add_argument("--threshold-b", type=float, help="Umbral explícito de B")
ap.add_argument("--metrics-b", type=Path, help="metrics.json del que leer el umbral de B")
ap.add_argument("--out", required=True, type=Path); ap.add_argument("--n-boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
def threshold(value, metrics_path, name):
    if (value is None) == (metrics_path is None):
        raise ValueError(f"Indique exactamente uno de --threshold-{name} o --metrics-{name}")
    return float(value if value is not None else json.loads(metrics_path.read_text())["threshold"])
threshold_a, threshold_b = threshold(a.threshold_a, a.metrics_a, "a"), threshold(a.threshold_b, a.metrics_b, "b")
keys = ["patient_id", "study_id", "label"]
left, right = pd.read_csv(a.model_a), pd.read_csv(a.model_b)
required = set(keys + ["score"])
for name, frame in (("model-a", left), ("model-b", right)):
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} no tiene columnas requeridas: {sorted(missing)}")
    if frame.duplicated(keys).any():
        raise ValueError(f"{name} tiene claves de imagen duplicadas")
merged = left[keys + ["score"]].merge(right[keys + ["score"]], on=keys, how="inner", suffixes=("_a", "_b"), validate="one_to_one")
if len(merged) != len(left) or len(merged) != len(right): raise ValueError("Las predicciones no cubren exactamente las mismas imágenes")
y, pa = merged.label.to_numpy(), merged.patient_id.to_numpy()
out = {"model_a": metrics(y, merged.score_a, threshold_a), "model_b": metrics(y, merged.score_b, threshold_b), "difference_a_minus_b_bootstrap_95": paired_patient_bootstrap_difference(y, merged.score_a, threshold_a, merged.score_b, threshold_b, pa, a.n_boot, a.seed), "unit": "paciente", "paired_images": len(merged), "paired_patients": int(merged.patient_id.nunique()), "interpretation": "La evidencia comparativa es el IC bootstrap pareado de A - B; no se infiere significancia por solapamiento de IC marginales."}
a.out.parent.mkdir(parents=True, exist_ok=True); write_json(a.out, out); print(json.dumps(out, indent=2))
