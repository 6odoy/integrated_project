"""Genera las figuras que consume el informe de la primera entrega."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROYECTO = Path(__file__).resolve().parents[1]
SALIDA_POR_DEFECTO = PROYECTO.parent / "report" / "figures"

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument(
    "--output-dir", type=Path, default=SALIDA_POR_DEFECTO,
    help="Directorio de figuras (por defecto, first-installment/report/figures).",
)
a = ap.parse_args()
a.output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- Figura 1
# OJO: solo DenseNet201 esta documentado en el texto de la entrega (seccion 1.3).
# Las demas filas hay que COMPLETARLAS a mano desde la Tabla 4 de Rahman et al.
# El script grafica unicamente lo que este relleno; no inventa valores.
CSV = PROYECTO / "data" / "rahman_tabla4.csv"
if not CSV.exists():
    pd.DataFrame({
        "red":          ["AlexNet", "ResNet18", "DenseNet201", "SqueezeNet"],
        "precision":    [np.nan, np.nan, 98.0, np.nan],
        "sensibilidad": [np.nan, np.nan, 99.0, np.nan],
        "especificidad":[np.nan, np.nan, 97.0, np.nan],
        "f1":           [np.nan, np.nan, 98.1, np.nan],
    }).to_csv(CSV, index=False)
    print(f"  plantilla creada: {CSV}  <-- COMPLETAR desde la Tabla 4 del articulo")

t = pd.read_csv(CSV)
metricas = ["precision", "sensibilidad", "especificidad", "f1"]
etiq = ["Precision", "Sensibilidad", "Especificidad", "F1"]
x = np.arange(len(metricas)); ancho = 0.8 / len(t)

fig, ax = plt.subplots(figsize=(8, 4.2))
for i, (_, r) in enumerate(t.iterrows()):
    vals = [r[m] for m in metricas]
    if all(pd.isna(v) for v in vals):
        continue
    b = ax.bar(x + i * ancho - 0.4 + ancho / 2, [0 if pd.isna(v) else v for v in vals],
               ancho, label=r["red"])
    for rect, v in zip(b, vals):
        if not pd.isna(v):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.6, f"{v:.1f}",
                    ha="center", fontsize=7)

ax.set_xticks(x); ax.set_xticklabels(etiq); ax.set_ylim(80, 102)
ax.set_ylabel("Porcentaje (%)")
ax.set_title("Metricas reportadas por Rahman et al. (2020) - Normal vs. Neumonia")
ax.legend(fontsize=8, ncol=4, loc="lower right"); ax.grid(axis="y", alpha=0.3)
faltan = t[t[metricas].isna().all(axis=1)]["red"].tolist()
if faltan:
    ax.text(0.01, 0.02, "Pendiente completar: " + ", ".join(faltan),
            transform=ax.transAxes, fontsize=7, color="crimson")
destino = a.output_dir / "figura1.png"
fig.tight_layout(); fig.savefig(destino, dpi=200); plt.close(fig)
print(f"  {destino}")

# ---------------------------------------------------------------- Figura 2
fases = [
    ("Fase 1", "Comprension del articulo\nProblema local\nLinea base\n\nFIJA la metrica:\nsensibilidad @ esp>=0,80"),
    ("Fase 2", "Transfer learning\nDenseNet201 + ImageNet\n(a) extractor congelado\n(b) ultimo bloque denso\n3 semillas"),
    ("Fase 3", "Grad-CAM y auditoria\nAnalisis de fallos\n(etiquetas -1)\n\nDecision: se justifica\nla profundidad?"),
]
fig, ax = plt.subplots(figsize=(10, 3.6)); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 3.6)
for i, (tit, cuerpo) in enumerate(fases):
    xc = 0.3 + i * 3.35
    ax.add_patch(FancyBboxPatch((xc, 0.5), 2.9, 2.5, boxstyle="round,pad=0.06",
                                lw=1.4, ec="#333", fc="#f2f5f9"))
    ax.text(xc + 1.45, 2.72, tit, ha="center", fontweight="bold", fontsize=11)
    ax.text(xc + 1.45, 1.65, cuerpo, ha="center", va="center", fontsize=8)
    if i < 2:
        ax.add_patch(FancyArrowPatch((xc + 2.95, 1.75), (xc + 3.28, 1.75),
                                     arrowstyle="-|>", mutation_scale=16, lw=1.4, color="#333"))
ax.text(5.0, 0.16, "El criterio de exito se fija en la Fase 1 y no cambia despues: "
                   "ninguna decision posterior puede modificarlo.",
        ha="center", fontsize=8, style="italic")
destino = a.output_dir / "figura2.png"
fig.tight_layout(); fig.savefig(destino, dpi=200); plt.close(fig)
print(f"  {destino}")
