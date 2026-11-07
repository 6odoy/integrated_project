"""Elige el modelo final por validación y ejecuta la auditoría de Fase 3.

Regla de selección, fijada antes de ver external_test: la configuración de
adaptación con mayor AUC medio en NIH adapt_val entre las tres semillas; dentro
de ella se audita la semilla 42. external_test no participa en la elección.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIGURATIONS = ("frozen", "finetune_partial", "imagenet_finetune_partial")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partition", required=True, type=Path)
    ap.add_argument("--images-root", required=True, type=Path)
    ap.add_argument("--bbox", required=True, type=Path)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--kermany-results", required=True, type=Path)
    ap.add_argument("--kermany-partition", required=True, type=Path)
    ap.add_argument("--kermany-images", required=True, type=Path)
    ap.add_argument("--figures", required=True, type=Path)
    ap.add_argument("--audit-seed", type=int, default=42)
    args = ap.parse_args()
    validation = {}
    for name in CONFIGURATIONS:
        runs = sorted((args.results / "adapt" / name).glob("seed*/metrics.json"))
        if runs:
            aucs = [json.loads(p.read_text())["best_val_auc"] for p in runs]
            validation[name] = {"mean_val_auc": sum(aucs) / len(aucs), "seeds": len(aucs)}
    if not validation:
        raise SystemExit("No hay corridas de adaptación; ejecute primero run_phase3.py")
    chosen = max(validation, key=lambda k: validation[k]["mean_val_auc"])
    final_run = args.results / "adapt" / chosen / f"seed{args.audit_seed}"
    # El Grad-CAM de referencia es el modelo Kermany del que partió el final (ImageNet no tiene uno).
    source_run = args.kermany_results / "finetune_partial" / f"seed{args.audit_seed}"
    audit = args.results / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    selection = {"rule": "max mean best_val_auc on NIH adapt_val across seeds", "validation": validation,
                 "chosen_configuration": chosen, "audited_run": str(final_run), "source_run": str(source_run)}
    (audit / "final_model.json").write_text(json.dumps(selection, indent=2), encoding="utf8")
    print(json.dumps(selection, indent=2), flush=True)
    common = ["--partition", args.partition, "--images-root", args.images_root, "--bbox", args.bbox,
              "--final-run", final_run]
    steps = [
        ("audit_gradcam.py", common + ["--source-run", source_run, "--out", audit, "--figures", args.figures]),
        ("audit_tests.py", common + ["--kermany-partition", args.kermany_partition,
                                     "--kermany-images", args.kermany_images, "--out", audit / "known_response_tests.json"]),
        ("audit_failures.py", ["--final-run", final_run, "--zero-shot-run",
                               args.results / "zero_shot" / "finetune_partial" / f"seed{args.audit_seed}",
                               "--out", audit, "--figures", args.figures]),
    ]
    for script, arguments in steps:
        print(f"[auditoría] {script}", flush=True)
        subprocess.run([sys.executable, str(HERE / script), *map(str, arguments)], check=True)


if __name__ == "__main__":
    main()
