"""Aplica un checkpoint Kermany a BRAX, sin reentrenar ni reoptimizar umbral."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
sys.path.insert(0, str(Path(__file__).parent))
from train import XrayDataset, build_model, predict
from common import metrics, patient_bootstrap, write_json

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True, type=Path); ap.add_argument("--images-root", required=True, type=Path)
ap.add_argument("--run-dir", required=True, type=Path); ap.add_argument("--out", required=True, type=Path); ap.add_argument("--n-boot", type=int, default=1000)
a = ap.parse_args(); run = json.loads((a.run_dir / "run_config.json").read_text()); frozen = json.loads((a.run_dir / "metrics.json").read_text())["threshold"]
df = pd.read_csv(a.manifest); device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
loader=DataLoader(XrayDataset(df,a.images_root,tf),batch_size=run["batch_size"],shuffle=False,num_workers=0)
# El checkpoint ya contiene todos los pesos; no se vuelve a descargar ImageNet.
model=build_model(run["mode"],False).to(device); model.load_state_dict(torch.load(a.run_dir / "best_checkpoint.pt",map_location=device,weights_only=True))
y,s,p,study=predict(model,loader,device); output=metrics(y,s,frozen); output.update({"threshold_frozen_from_kermany_validation":frozen,"prevalence":float(y.mean()),"patients":int(df.patient_id.nunique()),"images":len(df),"source_run":str(a.run_dir),"external_data_used_for_decisions":False,"bootstrap_95":patient_bootstrap(y,s,p,frozen,a.n_boot,run["seed"])})
a.out.mkdir(parents=True,exist_ok=True); pd.DataFrame({"patient_id":p,"study_id":study,"label":y,"score":s,"prediction":(s>=frozen).astype(int)}).to_csv(a.out / "predictions_brax.csv",index=False); write_json(a.out / "metrics_brax.json",output); print(json.dumps(output,indent=2))
