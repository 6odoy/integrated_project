"""Ajusta un checkpoint Kermany con BRAX y evalúa una vez en external_test.

La parada temprana y el umbral usan exclusivamente adapt_val. Ninguna imagen de
external_test interviene en pesos, selección de época, umbral o configuración.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from train import XrayDataset, build_model, predict, seed_everything
from common import metrics, patient_bootstrap, select_threshold, write_json


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--images-root", required=True, type=Path)
    ap.add_argument("--source-run", required=True, type=Path, help="Corrida Kermany con best_checkpoint.pt")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mode", choices=("frozen", "finetune_partial"), default="finetune_partial",
                    help="Alternativa de transferencia a comparar en BRAX")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=1000)
    return ap.parse_args()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def require_valid_partition(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    required_splits = {"adapt_train", "adapt_val", "external_test"}
    required_columns = {"patient_id", "study_id", "image_path", "label", "split"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Manifiesto incompleto; faltan columnas: {sorted(missing)}")
    present = set(frame.split.dropna().unique())
    if present != required_splits:
        raise ValueError("Se requiere el manifiesto particionado por prepare_brax.py")
    counts = frame.groupby("patient_id").split.nunique()
    if (counts > 1).any():
        raise RuntimeError("LEAKAGE: al menos un paciente pertenece a más de un split")
    splits = {name: frame.loc[frame.split == name].copy() for name in required_splits}
    for name, subset in splits.items():
        if subset.empty or subset.label.nunique() != 2:
            raise ValueError(f"{name} debe contener imágenes de ambas clases")
    return splits


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text())
    source_config_path, checkpoint = args.source_run / "run_config.json", args.source_run / "best_checkpoint.pt"
    if not source_config_path.exists() or not checkpoint.exists():
        raise FileNotFoundError("source-run debe contener run_config.json y best_checkpoint.pt")
    source = json.loads(source_config_path.read_text())
    seed_everything(args.seed)
    splits = require_valid_partition(pd.read_csv(args.manifest))
    device = get_device()
    image_size = cfg["image_size"]
    train_tf = transforms.Compose([transforms.Resize((image_size, image_size)),
        transforms.RandomAffine(10, translate=(.05, .05), scale=(.95, 1.05)), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    eval_tf = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {name: DataLoader(XrayDataset(frame, args.images_root, train_tf if name == "adapt_train" else eval_tf),
                                batch_size=cfg["batch_size"], shuffle=name == "adapt_train", num_workers=0,
                                generator=generator if name == "adapt_train" else None)
               for name, frame in splits.items()}
    model = build_model("finetune_partial", False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    # En ambos modos se aprende la cabeza; el modo parcial también ajusta denseblock4/norm5.
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    if args.mode == "finetune_partial":
        for parameter in model.features.denseblock4.parameters():
            parameter.requires_grad = True
        for parameter in model.features.norm5.parameters():
            parameter.requires_grad = True
    head = list(model.classifier.parameters())
    backbone = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("classifier")]
    parameter_groups = [{"params": head, "lr": cfg["head_learning_rate"]}]
    if backbone:
        parameter_groups.append({"params": backbone, "lr": cfg["backbone_learning_rate"]})
    optimizer = torch.optim.Adam(parameter_groups, weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                             patience=cfg["scheduler_patience"], factor=.1)
    positive_weight = torch.tensor([(splits["adapt_train"].label == 0).sum() /
                                    max((splits["adapt_train"].label == 1).sum(), 1)],
                                   device=device, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    args.out.mkdir(parents=True, exist_ok=True)
    best_auc, stale, best_epoch, history = -np.inf, 0, 0, []
    started = time.perf_counter()
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        losses, gradient_norms = [], []
        for images, labels, _, _ in loaders["adapt_train"]:
            optimizer.zero_grad()
            loss = loss_fn(model(images.to(device)).flatten(), labels.float().to(device))
            loss.backward()
            gradient_norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))))
            optimizer.step()
            losses.append(loss.item())
        y_val, scores_val, _, _ = predict(model, loaders["adapt_val"], device)
        val_auc = float(roc_auc_score(y_val, scores_val))
        scheduler.step(val_auc)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_auc": val_auc,
                        "lr_head": optimizer.param_groups[0]["lr"],
                        "lr_backbone": optimizer.param_groups[1]["lr"] if backbone else 0.0,
                        "gradient_norm": float(np.mean(gradient_norms))})
        if val_auc > best_auc:
            best_auc, stale, best_epoch = val_auc, 0, epoch
            torch.save(model.state_dict(), args.out / "best_checkpoint.pt")
        else:
            stale += 1
        if stale >= cfg["early_stopping_patience"]:
            break
    model.load_state_dict(torch.load(args.out / "best_checkpoint.pt", map_location=device, weights_only=True))
    y_val, scores_val, _, _ = predict(model, loaders["adapt_val"], device)
    threshold = select_threshold(y_val, scores_val, cfg["specificity_minimum"])
    y_test, scores_test, patients, studies = predict(model, loaders["external_test"], device)
    result = metrics(y_test, scores_test, threshold)
    result.update({"seed": args.seed, "mode": args.mode, "source_run": str(args.source_run), "source_mode": source.get("mode"),
                   "device": str(device), "best_epoch": best_epoch, "best_val_auc": best_auc,
                   "threshold_from": "BRAX adapt_val", "threshold": threshold,
                   "training_data": "BRAX adapt_train only", "validation_data": "BRAX adapt_val only",
                   "test_data": "BRAX external_test only", "patients": int(splits["external_test"].patient_id.nunique()),
                   "images": len(splits["external_test"]), "prevalence": float(y_test.mean()),
                   "class_positive_weight": float(positive_weight), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "bootstrap_95": patient_bootstrap(y_test, scores_test, patients, threshold, args.n_boot, args.seed),
                   "elapsed_seconds": time.perf_counter() - started})
    pd.DataFrame(history).to_csv(args.out / "history.csv", index=False)
    pd.DataFrame({"patient_id": patients, "study_id": studies, "label": y_test, "score": scores_test,
                  "prediction": (scores_test >= threshold).astype(int)}).to_csv(
                      args.out / "predictions_brax_external_test.csv", index=False)
    write_json(args.out / "metrics_brax_adapted.json", result)
    write_json(args.out / "run_config.json", {**cfg, "seed": args.seed, "mode": args.mode, "source_run": str(args.source_run),
               "source_mode": source.get("mode"), "training_data": "BRAX adapt_train only",
               "validation_data": "BRAX adapt_val only", "test_data": "BRAX external_test only"})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
