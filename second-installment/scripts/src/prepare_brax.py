"""Normaliza BRAX para evaluación externa, sin tocar ningún modelo.

Se deben indicar las columnas del archivo de metadatos de la descarga oficial.
La clase negativa exige ``No Finding=1`` y ``Pneumonia=0``; la positiva exige
``Pneumonia=1``. Cualquier -1/NA queda excluido y se informa el flujo.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--metadata", required=True, type=Path); ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--patient-col", default="patient_id"); ap.add_argument("--study-col", default="study_id")
ap.add_argument("--path-col", default="image_path"); ap.add_argument("--view-col", default="view")
ap.add_argument("--pneumonia-col", default="Pneumonia"); ap.add_argument("--no-finding-col", default="No Finding")
a = ap.parse_args(); raw = pd.read_csv(a.metadata); log = [{"stage":"raw", "images":len(raw), "patients":raw[a.patient_col].nunique()}]
front = raw[raw[a.view_col].astype(str).str.upper().isin({"PA","AP"})].copy(); log.append({"stage":"frontal_PA_AP", "images":len(front), "patients":front[a.patient_col].nunique()})
p, n = front[a.pneumonia_col], front[a.no_finding_col]
eligible = front[(p.isin([0,1])) & (n.isin([0,1])) & (((p == 1) & (n == 0)) | ((p == 0) & (n == 1)))].copy()
eligible["label"] = eligible[a.pneumonia_col].astype(int)
out = eligible.rename(columns={a.patient_col:"patient_id",a.study_col:"study_id",a.path_col:"image_path",a.view_col:"view"})[["patient_id","study_id","image_path","view","label"]]
log.append({"stage":"binary_eligible", "images":len(out), "patients":out.patient_id.nunique(), "prevalence":out.label.mean()})
a.out.parent.mkdir(parents=True, exist_ok=True); out.to_csv(a.out, index=False); pd.DataFrame(log).to_csv(a.out.with_name("brax_filter_flow.csv"), index=False)
print(pd.DataFrame(log).to_string(index=False))
