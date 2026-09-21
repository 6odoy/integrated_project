"""Entrena una corrida de transferencia y congela el umbral en validación.

BRAX no se importa aquí: este programa solo ve Kermany train/val/test.
"""
from __future__ import annotations
import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score
from common import metrics, patient_bootstrap, select_threshold, write_json

class XrayDataset(Dataset):
    def __init__(self, frame, root, transform): self.frame, self.root, self.transform = frame.reset_index(drop=True), Path(root), transform
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = Image.open(self.root / row.image_path).convert("RGB")
        return self.transform(image), int(row.label), str(row.patient_id), str(row.study_id)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

def build_model(mode, pretrained=True):
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    for parameter in model.features.parameters(): parameter.requires_grad = False
    if mode == "finetune_partial":
        for parameter in model.features.denseblock4.parameters(): parameter.requires_grad = True
        for parameter in model.features.norm5.parameters(): parameter.requires_grad = True
    elif mode != "frozen": raise ValueError("mode debe ser frozen o finetune_partial")
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model

@torch.inference_mode()
def predict(model, loader, device):
    model.eval(); scores, labels, patients, studies = [], [], [], []
    for x, y, p, s in loader:
        scores.extend(torch.sigmoid(model(x.to(device))).flatten().cpu().numpy())
        labels.extend(y.numpy()); patients.extend(p); studies.extend(s)
    return np.asarray(labels), np.asarray(scores), np.asarray(patients), np.asarray(studies)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True, type=Path); ap.add_argument("--images-root", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path); ap.add_argument("--mode", choices=("frozen", "finetune_partial"), required=True)
    ap.add_argument("--seed", required=True, type=int); ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-boot", type=int, default=1000); ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args(); cfg = json.loads(args.config.read_text()); seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.partition); splits = {s: df[df.split == s] for s in ("train", "val", "test")}
    train_tf = transforms.Compose([transforms.Resize((224,224)), transforms.RandomAffine(10, translate=(.05,.05), scale=(.95,1.05)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    eval_tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    loaders = {s: DataLoader(XrayDataset(splits[s], args.images_root, train_tf if s == "train" else eval_tf), batch_size=cfg["batch_size"], shuffle=s == "train", num_workers=0) for s in splits}
    model = build_model(args.mode, not args.no_pretrained).to(device)
    positive_weight = torch.tensor([(splits["train"].label == 0).sum() / max((splits["train"].label == 1).sum(), 1)], device=device, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    head = list(model.classifier.parameters()); backbone = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith("classifier")]
    groups = [{"params": head, "lr": cfg["head_learning_rate"]}] + ([{"params": backbone, "lr": cfg["backbone_learning_rate"]}] if backbone else [])
    optimizer = torch.optim.Adam(groups, weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=cfg["scheduler_patience"], factor=.1)
    history, best_auc, best_epoch, stale = [], -np.inf, 0, 0; started = time.perf_counter()
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train(); losses, grad_norms = [], []
        for x,y,_,_ in loaders["train"]:
            optimizer.zero_grad(); logits = model(x.to(device)).flatten(); loss = loss_fn(logits, y.float().to(device)); loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf")); optimizer.step()
            losses.append(loss.item()); grad_norms.append(float(grad_norm))
        ytr, str_, _, _ = predict(model, loaders["train"], device); yv, sv, _, _ = predict(model, loaders["val"], device)
        train_auc, val_auc = roc_auc_score(ytr,str_), roc_auc_score(yv,sv); scheduler.step(val_auc)
        history.append({"epoch":epoch, "train_loss":float(np.mean(losses)), "train_auc":train_auc, "val_auc":val_auc, "lr_head":optimizer.param_groups[0]["lr"], "gradient_norm":float(np.mean(grad_norms))})
        if val_auc > best_auc:
            best_auc, best_epoch, stale = val_auc, epoch, 0; torch.save(model.state_dict(), args.out / "best_checkpoint.pt")
        else: stale += 1
        if stale >= cfg["early_stopping_patience"]: break
    model.load_state_dict(torch.load(args.out / "best_checkpoint.pt", map_location=device, weights_only=True))
    yv, sv, _, _ = predict(model, loaders["val"], device); threshold = select_threshold(yv, sv, cfg["specificity_minimum"])
    yt, st, patients, studies = predict(model, loaders["test"], device)
    warm = next(iter(loaders["test"]))[0][:1].to(device); model.eval()
    with torch.inference_mode(): model(warm); tic=time.perf_counter(); [model(warm) for _ in range(20)]
    inference_ms = (time.perf_counter()-tic)*1000/20
    result = metrics(yt, st, threshold); result.update({"seed":args.seed, "mode":args.mode, "threshold":threshold, "best_epoch":best_epoch, "best_val_auc":best_auc, "inference_ms_image":inference_ms, "elapsed_seconds":time.perf_counter()-started, "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad), "device":str(device), "prevalence":float(yt.mean())})
    result["bootstrap_95"] = patient_bootstrap(yt, st, patients, threshold, args.n_boot, args.seed)
    pd.DataFrame(history).to_csv(args.out / "history.csv", index=False)
    pd.DataFrame({"patient_id":patients,"study_id":studies,"label":yt,"score":st,"prediction":(st>=threshold).astype(int)}).to_csv(args.out / "predictions_kermany_test.csv", index=False)
    write_json(args.out / "metrics.json", result)
    write_json(args.out / "run_config.json", {**cfg, "seed":args.seed, "mode":args.mode, "pretrained":not args.no_pretrained, "class_positive_weight":float(positive_weight), "dataset":"Kermany only; BRAX not used", "torch":torch.__version__})
    print(json.dumps(result, indent=2))
if __name__ == "__main__": main()
