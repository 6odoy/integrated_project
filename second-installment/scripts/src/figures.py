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
# IC95% bootstrap por paciente: se preservan por corrida, sin promediar límites.
ci_lines=["Configuración,semilla,sensibilidad_IC95,especificidad_IC95,auc_roc_IC95,auprc_IC95"]
def ci(value, interval):
    return "--" if not isinstance(interval,(list,tuple)) or len(interval)!=2 else f"{value:.3f} [{interval[0]:.3f}, {interval[1]:.3f}]"
for d in sorted(runs,key=lambda x:(x["mode"],x["seed"])):
    boot=d.get("bootstrap_95",{}); name="Congelado" if d["mode"]=="frozen" else "Fine-tuning parcial"
    ci_lines.append(",".join([name,str(d["seed"]),ci(d["sensitivity"],boot.get("sensitivity")),ci(d["specificity"],boot.get("specificity")),ci(d["auc_roc"],boot.get("auc_roc")),ci(d["auprc"],boot.get("auprc"))]))
(a.generated/"bootstrap_95_by_run.csv").write_text("\n".join(ci_lines)+"\n")

# Suma de semillas: inspección de FN/FP, explícitamente no un modelo adicional.
confusions={}
for d in runs:
    pth=a.results/d["run_dir"] / "predictions_kermany_test.csv"
    if pth.exists():
        p=pd.read_csv(pth); cm=np.array([[((p.label==0)&(p.prediction==0)).sum(),((p.label==0)&(p.prediction==1)).sum()],[((p.label==1)&(p.prediction==0)).sum(),((p.label==1)&(p.prediction==1)).sum()]])
        confusions[d["mode"]]=confusions.get(d["mode"],np.zeros((2,2),dtype=int))+cm
cm_rows=[]
for mode,cm in sorted(confusions.items()):
    for actual,row in zip(("negativa","positiva"),cm): cm_rows.append({"configuracion":mode,"real":actual,"pred_negativa":row[0],"pred_positiva":row[1]})
pd.DataFrame(cm_rows).to_csv(a.generated/"confusion_matrices.csv",index=False)

fig,axes=plt.subplots(2,2,figsize=(10,6.4),sharex="col"); legacy_loss=False
for d in runs:
    h=pd.read_csv(a.results/d["run_dir"]/"history.csv"); label=("Congelado" if d["mode"]=="frozen" else "Fine-tuning")+f" s{d['seed']}"
    axes[0,0].plot(h.epoch,h.train_loss,label=label)
    if "val_loss" in h: axes[0,0].plot(h.epoch,h.val_loss,linestyle="--",color=axes[0,0].lines[-1].get_color())
    else: legacy_loss=True
    axes[0,1].plot(h.epoch,h.train_auc,label=label); axes[0,1].plot(h.epoch,h.val_auc,linestyle="--",color=axes[0,1].lines[-1].get_color())
    axes[1,0].plot(h.epoch,h.lr_head,label=label)
    axes[1,1].plot(h.epoch,h.gradient_norm,label=label)
axes[0,0].set(title="Pérdida BCE ponderada",ylabel="Pérdida")
axes[0,1].set(title="AUC (continua: train; discontinua: val)",ylabel="AUC")
axes[1,0].set(title="Tasa de aprendizaje — cabeza",xlabel="Época",ylabel="LR",yscale="log")
axes[1,1].set(title="Norma media del gradiente",xlabel="Época",ylabel="Norma",yscale="log")
if legacy_loss: axes[0,0].text(.01,.02,"Historial legado: sin pérdida de validación",transform=axes[0,0].transAxes,fontsize=7)
for ax in axes.flat: ax.grid(alpha=.25); ax.legend(fontsize=6,ncol=2)
fig.tight_layout(); fig.savefig(a.figures/"training_curves.png",dpi=200); plt.close(fig)

if confusions:
    fig,axes=plt.subplots(1,len(confusions),figsize=(4.1*len(confusions),3.5),squeeze=False)
    for ax,(mode,cm) in zip(axes.flat,sorted(confusions.items())):
        ax.imshow(cm,cmap="Blues")
        for (i,j),value in np.ndenumerate(cm): ax.text(j,i,str(value),ha="center",va="center",fontsize=11)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Negativa","Positiva"],yticklabels=["Negativa","Positiva"],xlabel="Predicción",ylabel="Etiqueta real",title="Congelado" if mode=="frozen" else "Fine-tuning parcial")
    fig.suptitle("Matriz acumulada de 3 semillas (conteos, no modelo adicional)",fontsize=10); fig.tight_layout(); fig.savefig(a.figures/"confusion_matrices.png",dpi=200); plt.close(fig)
