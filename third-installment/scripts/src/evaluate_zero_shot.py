"""Aplica un checkpoint Kermany a NIH sin reentrenar: mide la brecha de dominio.

Se reportan dos puntos de operación sobre el mismo external_test:

* ``kermany_threshold``: el umbral congelado en la validación de Kermany.
* ``nih_val_threshold``: el umbral reelegido en NIH adapt_val, sin tocar pesos.

La diferencia entre ambos separa la parte del desplazamiento que se corrige con
recalibrar el umbral de la que exige adaptar la representación.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nih_common import evaluate_split, load_partition, make_loader
from common import metrics, select_threshold, write_json
from train import build_model, get_device, predict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partition", required=True, type=Path)
    ap.add_argument("--images-root", required=True, type=Path)
    ap.add_argument("--run-dir", required=True, type=Path, help="Corrida Kermany con best_checkpoint.pt y metrics.json")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    run = json.loads((args.run_dir / "run_config.json").read_text())
    kermany_threshold = json.loads((args.run_dir / "metrics.json").read_text())["threshold"]
    device = get_device()
    splits = load_partition(args.partition)
    model = build_model(run["mode"], pretrained=False)
    model.load_state_dict(torch.load(args.run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=True))
    model.to(device)
    val_loader = make_loader(splits["adapt_val"], args.images_root, run["image_size"], run["batch_size"])
    test_loader = make_loader(splits["external_test"], args.images_root, run["image_size"], run["batch_size"])
    y_val, s_val, _, _ = predict(model, val_loader, device)
    nih_threshold = select_threshold(y_val, s_val, run["specificity_minimum"])
    args.out.mkdir(parents=True, exist_ok=True)
    output = {"source_run": str(args.run_dir), "mode": run["mode"], "seed": run["seed"],
              "adapt_val_at_kermany_threshold": metrics(y_val, s_val, kermany_threshold),
              "weights_updated": False}
    for name, threshold in (("kermany_threshold", kermany_threshold), ("nih_val_threshold", nih_threshold)):
        result, predictions = evaluate_split(model, test_loader, splits["external_test"], threshold,
                                             device, args.n_boot, run["seed"])
        output[name] = {"threshold": threshold, **result}
        predictions.to_csv(args.out / f"predictions_nih_external_test_{name}.csv", index=False)
    write_json(args.out / "metrics.json", output)
    print(json.dumps({k: {m: v.get(m) for m in ("threshold", "sensitivity", "specificity", "auc_roc")}
                      for k, v in output.items() if k.endswith("threshold")}, indent=2))


if __name__ == "__main__":
    main()
