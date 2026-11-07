"""Traslada un checkpoint Kermany a NIH adulto y evalúa una vez en external_test.

Parada temprana y umbral usan exclusivamente adapt_val. Ninguna imagen de
external_test interviene en pesos, época, umbral o configuración.

``--source imagenet`` es el control: mismo esquema, pero desde los pesos de
ImageNet sin pasar por Kermany, para medir si la réplica aporta algo al traslado.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from nih_common import evaluate_split, load_partition, make_loader, subsample_patients
from common import select_threshold, write_json
from train import build_model, fit, get_device, measure_latency, predict, seed_everything, set_trainable


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partition", required=True, type=Path)
    ap.add_argument("--images-root", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--source", required=True, help="Corrida Kermany con best_checkpoint.pt, o 'imagenet'")
    ap.add_argument("--mode", choices=("frozen", "finetune_partial"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train-fraction", type=float, default=1.0, help="Fracción de pacientes de adapt_train")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    seed_everything(args.seed)
    device = get_device()
    splits = load_partition(args.partition)
    splits["adapt_train"] = subsample_patients(splits["adapt_train"], args.train_fraction, args.seed)
    started = time.perf_counter()
    loaders = {name: make_loader(frame, args.images_root, cfg["image_size"], cfg["batch_size"],
                                 train=name == "adapt_train", seed=args.seed)
               for name, frame in splits.items()}
    print(f"cache NIH listo en {time.perf_counter() - started:.0f}s "
          f"({len(splits['adapt_train'])} imágenes de adaptación)", flush=True)
    if args.source == "imagenet":
        model = build_model(args.mode, pretrained=True)
    else:
        model = build_model(args.mode, pretrained=False)
        model.load_state_dict(torch.load(Path(args.source) / "best_checkpoint.pt", map_location="cpu", weights_only=True))
        set_trainable(model, args.mode)
    model.to(device)
    train = splits["adapt_train"]
    positive_weight = float((train.label == 0).sum() / max((train.label == 1).sum(), 1))
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    history, best_epoch, best_auc = fit(model, loaders["adapt_train"], loaders["adapt_val"], cfg, device, args.out, positive_weight)
    training_seconds = time.perf_counter() - started
    y_val, s_val, _, _ = predict(model, loaders["adapt_val"], device)
    threshold = select_threshold(y_val, s_val, cfg["specificity_minimum"])
    result, predictions = evaluate_split(model, loaders["external_test"], splits["external_test"], threshold,
                                         device, args.n_boot, args.seed)
    result.update({
        "seed": args.seed, "mode": args.mode, "source": str(args.source), "train_fraction": args.train_fraction,
        "adapt_train_images": len(train), "adapt_train_patients": int(train.patient_id.nunique()),
        "threshold": threshold, "threshold_from": "NIH adapt_val", "best_epoch": best_epoch,
        "best_val_auc": best_auc, "epochs_run": len(history), "elapsed_seconds": training_seconds,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "class_positive_weight": positive_weight, "device": str(device),
        "prevalence": float(predictions.label.mean()), "images": len(predictions),
        "patients": int(predictions.patient_id.nunique()),
    })
    result.update(measure_latency(model, next(iter(loaders["external_test"]))[0], device))
    history.to_csv(args.out / "history.csv", index=False)
    pd.DataFrame({"study_id": splits["adapt_val"].study_id, "label": y_val, "score": s_val}).to_csv(
        args.out / "predictions_nih_adapt_val.csv", index=False)
    predictions.to_csv(args.out / "predictions_nih_external_test.csv", index=False)
    write_json(args.out / "metrics.json", result)
    write_json(args.out / "run_config.json", {**cfg, "seed": args.seed, "mode": args.mode, "source": str(args.source),
               "train_fraction": args.train_fraction, "frozen_batchnorm_in_eval": True,
               "training_data": "NIH adapt_train only", "validation_data": "NIH adapt_val only",
               "test_data": "NIH external_test only", "torch": torch.__version__})
    print(json.dumps({k: v for k, v in result.items() if k != "bootstrap_95"}, indent=2, default=str))


if __name__ == "__main__":
    main()
