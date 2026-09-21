"""Construye el manifiesto y una partición Kermany reproducible para Fase 2."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import split_by_patient

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True, type=Path)
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
df = pd.read_csv(a.manifest)
partition = split_by_patient(df, a.seed)
a.out.mkdir(parents=True, exist_ok=True)
partition.to_csv(a.out / "partition_kermany.csv", index=False)
summary = partition.groupby("split").agg(images=("label", "size"), patients=("patient_id", "nunique"), prevalence=("label", "mean"))
summary.to_csv(a.out / "partition_summary.csv")
print(summary.to_string())
print(f"Sin fuga: {partition.patient_id.nunique()} pacientes asignados una sola vez.")
