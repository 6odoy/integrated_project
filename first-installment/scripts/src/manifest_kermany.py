"""Construye data/manifest_kermany.csv desde la estructura de carpetas de Kermany.

Estructura tras descomprimir ChestXRay2017.zip:
    chest_xray/{train,test}/{NORMAL,PNEUMONIA}/*.jpeg

Nombres de archivo:
    PNEUMONIA -> person1_bacteria_1.jpeg  /  person1_virus_6.jpeg
    NORMAL    -> IM-0115-0001.jpeg  /  NORMAL2-IM-1427-0001.jpeg

LIMITACIONES QUE HAY QUE DECLARAR EN LA ENTREGA
-----------------------------------------------
1. El paciente solo es recuperable en los positivos (prefijo personN). En los
   NORMAL no hay identificador de paciente: cada archivo se trata como un
   paciente distinto, lo que puede subestimar el agrupamiento.
2. La numeracion personN se reinicia entre train/ y test/, asi que el id se
   prefija con la carpeta de origen para no fusionar pacientes distintos.
3. No hay etiquetas de vista: todas son frontales pediatricas. Se marca AP.
4. No hay etiquetas de incertidumbre (-1): esa rama del pipeline queda sin
   ejercitar hasta que llegue BRAX.
5. La prevalencia esta enriquecida artificialmente (~74 % neumonia en train).
   NO es la prevalencia real de un servicio de urgencias.
"""
from __future__ import annotations
import argparse, os, re, sys
from pathlib import Path
import pandas as pd

RE_PERSONA = re.compile(r"person(\d+)", re.I)

PROYECTO = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=PROYECTO / "data" / "chest_xray")
ap.add_argument("--out", default=PROYECTO / "data" / "manifest_kermany.csv")
a = ap.parse_args()

if not os.path.isdir(a.root):
    sys.exit(f"No existe {a.root}. Descomprime ChestXRay2017.zip primero.")

filas = []
for origen in ("train", "test", "val"):
    for clase, etiqueta in (("NORMAL", 0), ("PNEUMONIA", 1)):
        d = os.path.join(a.root, origen, clase)
        if not os.path.isdir(d):
            continue
        for nombre in sorted(os.listdir(d)):
            if not nombre.lower().endswith((".jpeg", ".jpg", ".png")):
                continue
            m = RE_PERSONA.search(nombre)
            # id prefijado con la carpeta: personN se reinicia entre train y test
            pid = f"{origen}_person{m.group(1)}" if m else f"{origen}_{os.path.splitext(nombre)[0]}"
            filas.append(dict(
                patient_id=pid,
                study_id=os.path.splitext(nombre)[0],
                image_path=os.path.join(origen, clase, nombre),
                view="AP",
                label=etiqueta,
                origen_kermany=origen,     # se ignora: reparticionamos nosotros
            ))

df = pd.DataFrame(filas)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)

print(f"{len(df)} imagenes / {df.patient_id.nunique()} pacientes -> {a.out}")
print(f"prevalencia global: {df.label.mean():.4f}")
print(df.groupby(["origen_kermany", "label"]).size().rename("n"))
multi = df.groupby("patient_id").size()
print(f"pacientes con >1 imagen: {(multi > 1).sum()}  (max {multi.max()} imagenes)")
