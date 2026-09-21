"""Orquesta las 2 configuraciones x 3 semillas sin usar BRAX."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument("--partition",required=True); ap.add_argument("--images-root",required=True); ap.add_argument("--config",required=True); ap.add_argument("--results",required=True); ap.add_argument("--n-boot",type=int,default=1000); ap.add_argument("--no-pretrained",action="store_true")
a=ap.parse_args(); cfg=json.loads(Path(a.config).read_text())
for mode in ("frozen","finetune_partial"):
 for seed in cfg["seeds"]:
    out=Path(a.results)/"kermany"/mode/f"seed{seed}"; command=[sys.executable,str(Path(__file__).with_name("train.py")),"--partition",a.partition,"--images-root",a.images_root,"--config",a.config,"--mode",mode,"--seed",str(seed),"--out",str(out),"--n-boot",str(a.n_boot)]
    if a.no_pretrained: command.append("--no-pretrained")
    subprocess.run(command,check=True)
