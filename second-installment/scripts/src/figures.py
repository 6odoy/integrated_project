"""Genera figuras y tablas LaTeX exclusivamente desde resultados guardados."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

ap=argparse.ArgumentParser(); ap.add_argument("--results",type=Path,required=True); ap.add_argument("--figures",type=Path,required=True); ap.add_argument("--generated",type=Path,required=True)
a=ap.parse_args(); a.figures.mkdir(parents=True,exist_ok=True); a.generated.mkdir(parents=True,exist_ok=True)
runs=[]
for path in a.results.glob("kermany/*/seed*/metrics.json"):
    d=json.loads(path.read_text()); d["run_dir"]=path.parent.relative_to(a.results); runs.append(d)
pd.DataFrame(runs).to_csv(a.generated / "all_runs.csv",index=False)
if not runs:
    (a.generated / "table_results.tex").write_text("\\begin{tabular}{lp{9cm}}\\toprule Estado & Evidencia \\\\ \\midrule Pendiente & Ejecute \\texttt{make phase2-train}; no se reportan métricas sin una corrida finalizada. \\\\ \\bottomrule\\end{tabular}\n")
    fig,ax=plt.subplots(figsize=(7,2.2)); ax.axis("off"); ax.text(.5,.5,"Curvas pendientes: no hay corridas finalizadas",ha="center",va="center"); fig.savefig(a.figures/"training_curves.png",dpi=200,bbox_inches="tight"); plt.close(fig)
    raise SystemExit(0)
rows=[]
for mode, group in pd.DataFrame(runs).groupby("mode"):
    row={"configuration":"Extractor congelado" if mode=="frozen" else "Fine-tuning parcial"}
    for metric in ("sensitivity","specificity","auc_roc","auprc","f1","inference_ms_image"):
        row[metric]=group[metric].mean(); row[metric+"_sd"]=group[metric].std(ddof=0)
    rows.append(row)
summary=pd.DataFrame(rows); summary.to_csv(a.generated/"kermany_summary.csv",index=False)
def fmt(row,key): return f"{row[key]:.3f} $\\pm$ {row[key+'_sd']:.3f}".replace('.',',')
lines=["\\begin{tabular}{lccccc}","\\toprule","Configuración & Sens. & Espec. & AUC-ROC & AUPRC & F1 \\\\ ","\\midrule"]
for _, r in summary.iterrows():
    values = " & ".join(fmt(r, k) for k in ("sensitivity", "specificity", "auc_roc", "auprc", "f1"))
    lines.append(r.configuration + " & " + values + " " + chr(92) * 2)
lines += ["\\bottomrule","\\end{tabular}"]; (a.generated/"table_results.tex").write_text("\n".join(lines))
fig,axes=plt.subplots(1,2,figsize=(10,3.4))
for d in runs:
    h=pd.read_csv(a.results/d["run_dir"]/"history.csv"); label=("Congelado" if d["mode"]=="frozen" else "Fine-tuning")+f" s{d['seed']}"
    axes[0].plot(h.epoch,h.train_loss,label=label); axes[1].plot(h.epoch,h.val_auc,label=label)
axes[0].set(title="Pérdida de entrenamiento",xlabel="Época",ylabel="BCE ponderada"); axes[1].set(title="AUC de validación",xlabel="Época",ylabel="AUC")
for ax in axes: ax.grid(alpha=.25); ax.legend(fontsize=6,ncol=2)
fig.tight_layout(); fig.savefig(a.figures/"training_curves.png",dpi=200); plt.close(fig)
