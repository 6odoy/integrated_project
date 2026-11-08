"""Orquesta el traslado a NIH; omite toda corrida que ya tenga metrics.json.

Orden: evaluación directa de los seis modelos Kermany, adaptación (2 modos x 3
semillas desde Kermany, más el control ImageNet), curva de datos escasos,
Random Forest en la misma partición y comparaciones pareadas.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE2 = HERE.parents[2] / "second-installment" / "scripts"


def run(script: Path, *arguments, done: Path | None = None) -> None:
    if done is not None and done.exists():
        print(f"[omitida] {done.parent}", flush=True)
        return
    print(f"[ejecuta] {script.name} {' '.join(map(str, arguments))}", flush=True)
    subprocess.run([sys.executable, str(script), *map(str, arguments)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partition", required=True, type=Path)
    ap.add_argument("--images-root", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--kermany-results", required=True, type=Path, help="second-installment/scripts/results/kermany")
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--fractions", default="0.1,0.25,0.5")
    ap.add_argument("--stage", choices=("all", "zero_shot", "adapt", "curve", "baseline", "compare"), default="all")
    args = ap.parse_args()
    seeds = json.loads(args.config.read_text())["seeds"]
    common = ["--partition", args.partition, "--images-root", args.images_root]
    stage = lambda name: args.stage in ("all", name)

    if stage("zero_shot"):
        for mode in ("frozen", "finetune_partial"):
            for seed in seeds:
                out = args.results / "zero_shot" / mode / f"seed{seed}"
                run(HERE / "evaluate_zero_shot.py", *common, "--run-dir", args.kermany_results / mode / f"seed{seed}",
                    "--out", out, done=out / "metrics.json")
    if stage("adapt"):
        # El traslado parte siempre del fine-tuning parcial de Kermany de la misma semilla.
        for mode in ("frozen", "finetune_partial"):
            for seed in seeds:
                out = args.results / "adapt" / mode / f"seed{seed}"
                run(HERE / "adapt_nih.py", *common, "--config", args.config, "--mode", mode, "--seed", seed,
                    "--source", args.kermany_results / "finetune_partial" / f"seed{seed}", "--out", out,
                    done=out / "metrics.json")
        for seed in seeds:
            out = args.results / "adapt" / "imagenet_finetune_partial" / f"seed{seed}"
            run(HERE / "adapt_nih.py", *common, "--config", args.config, "--mode", "finetune_partial",
                "--seed", seed, "--source", "imagenet", "--out", out, done=out / "metrics.json")
    if stage("curve"):
        for fraction in args.fractions.split(","):
            for seed in seeds:
                out = args.results / "curve" / f"fraction{fraction}" / f"seed{seed}"
                run(HERE / "adapt_nih.py", *common, "--config", args.config, "--mode", "finetune_partial",
                    "--seed", seed, "--train-fraction", fraction,
                    "--source", args.kermany_results / "finetune_partial" / f"seed{seed}", "--out", out,
                    done=out / "metrics.json")
    if stage("baseline"):
        out = args.results / "baseline_random_forest"
        run(PHASE2 / "src" / "baseline_random_forest.py", *common, "--out", out, "--train-split", "adapt_train",
            "--val-split", "adapt_val", "--test-split", "external_test",
            "--predictions-name", "predictions_nih_external_test.csv", done=out / "metrics.json")
    if stage("compare"):
        pairs = [(f"adapt/{mode}/seed{seed}", "baseline_random_forest", f"{mode}_seed{seed}_vs_rf")
                 for mode in ("frozen", "finetune_partial", "imagenet_finetune_partial") for seed in seeds]
        pairs += [(f"adapt/finetune_partial/seed{seed}", f"adapt/frozen/seed{seed}", f"partial_vs_frozen_seed{seed}")
                  for seed in seeds]
        pairs += [(f"adapt/finetune_partial/seed{seed}", f"adapt/imagenet_finetune_partial/seed{seed}",
                   f"kermany_vs_imagenet_seed{seed}") for seed in seeds]
        for a, b, name in pairs:
            out = args.results / "paired" / f"{name}.json"
            run(PHASE2 / "src" / "compare_paired.py",
                "--model-a", args.results / a / "predictions_nih_external_test.csv", "--metrics-a", args.results / a / "metrics.json",
                "--model-b", args.results / b / "predictions_nih_external_test.csv", "--metrics-b", args.results / b / "metrics.json",
                "--out", out, done=out)


if __name__ == "__main__":
    main()
