"""Pruebas de respuesta conocida y entradas fuera de dominio para el modelo final.

Cada prueba fija de antemano qué debería pasar si el modelo usa la señal
pulmonar y no un atajo:

1. Ocluir la caja de neumonía del radiólogo debe bajar el puntaje mucho más
   que ocluir una caja del mismo tamaño en el lado contralateral.
2. Ocluir el tórax central (60 % de la imagen) debe llevar el AUC cerca de 0,5;
   ocluir solo el marco periférico (marcadores, texto) debe dejarlo casi igual.
3. Voltear horizontalmente no debe cambiar el AUC: la neumonía no tiene lado.
4. Entradas que no son radiografías (gris uniforme, ruido, imagen invertida,
   rotada 90°) no deberían recibir puntajes confiados de neumonía.
5. Las imágenes pediátricas de Kermany miden si la adaptación olvidó el origen.

Las oclusiones rellenan con la media de intensidad de cada imagen.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from audit_utils import cached_images, get_device, load_model, scores
from nih_common import load_partition
from common import write_json

NIH_SIZE = 1024


def occlude(images: np.ndarray, masks: np.ndarray) -> np.ndarray:
    out = images.copy()
    fill = images.reshape(len(images), -1).mean(axis=1).astype(np.uint8)
    for i in range(len(images)):
        out[i][masks[i]] = fill[i]
    return out


def region_masks(n: int, size: int, region: str) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    if region == "center":
        a, b = int(.2 * size), int(.8 * size)
        mask[a:b, a:b] = True
    elif region == "border":
        m = int(.1 * size)
        mask[:] = True
        mask[m:size - m, m:size - m] = False
    return np.broadcast_to(mask, (n, size, size))


def auc_and_rate(y, s, threshold):
    return {"auc_roc": float(roc_auc_score(y, s)), "positive_rate": float((s >= threshold).mean()),
            "sensitivity": float((s[y == 1] >= threshold).mean()), "specificity": float((s[y == 0] < threshold).mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partition", required=True, type=Path)
    ap.add_argument("--images-root", required=True, type=Path)
    ap.add_argument("--bbox", required=True, type=Path)
    ap.add_argument("--final-run", required=True, type=Path)
    ap.add_argument("--kermany-partition", required=True, type=Path)
    ap.add_argument("--kermany-images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    device = get_device()
    model, threshold = load_model(args.final_run, device)
    test = load_partition(args.partition)["external_test"]
    images = cached_images(test, args.images_root)
    n, size = len(images), images.shape[1]
    y = test.label.to_numpy()
    base = scores(model, images, device)
    report = {"threshold": threshold, "original": auc_and_rate(y, base, threshold)}

    # 1. Caja del radiólogo frente a la misma caja reflejada al otro hemitórax.
    bbox = pd.read_csv(args.bbox)
    bbox = bbox[bbox["Finding Label"] == "Pneumonia"].rename(columns={"Bbox [x": "x", "y": "y", "w": "w", "h]": "h"})
    rows = []
    for _, b in bbox.iterrows():
        match = np.flatnonzero(test.image_path.to_numpy() == b["Image Index"])
        if not len(match):
            continue
        i = match[0]
        x, yy, w, h = np.array([b.x, b.y, b.w, b.h]) * size / NIH_SIZE
        box, mirror = np.zeros((size, size), bool), np.zeros((size, size), bool)
        box[int(yy):int(np.ceil(yy + h)), int(x):int(np.ceil(x + w))] = True
        mx = size - (x + w)
        mirror[int(yy):int(np.ceil(yy + h)), int(mx):int(np.ceil(mx + w))] = True
        mirror &= ~box  # si la caja cruza la línea media, el control no ocluye lo mismo
        pair = scores(model, occlude(images[[i, i]], np.stack([box, mirror])), device)
        rows.append({"study_id": test.study_id.iloc[i], "original": base[i], "box_occluded": pair[0],
                     "mirror_occluded": pair[1]})
    occ = pd.DataFrame(rows).drop_duplicates("study_id")
    occ["drop_box"], occ["drop_mirror"] = occ.original - occ.box_occluded, occ.original - occ.mirror_occluded
    report["bbox_occlusion"] = {
        "images": len(occ), "median_drop_box": float(occ.drop_box.median()),
        "median_drop_mirror": float(occ.drop_mirror.median()),
        "fraction_box_drop_greater": float((occ.drop_box > occ.drop_mirror).mean()),
        "wilcoxon_p": float(wilcoxon(occ.drop_box, occ.drop_mirror).pvalue) if len(occ) > 5 else None,
        "detected_before": float((occ.original >= threshold).mean()),
        "detected_after_box_occlusion": float((occ.box_occluded >= threshold).mean()),
    }

    # 2-3. Oclusión central vs periférica y volteo.
    for region in ("center", "border"):
        report[f"{region}_occluded"] = auc_and_rate(y, scores(model, occlude(images, region_masks(n, size, region)), device), threshold)
    flipped = scores(model, images[:, :, ::-1], device)
    report["horizontal_flip"] = {**auc_and_rate(y, flipped, threshold),
                                 "median_abs_score_change": float(np.median(np.abs(flipped - base)))}

    # 4. Entradas que no son radiografías: se reporta la tasa de "positivo".
    rng = np.random.default_rng(42)
    negatives = images[y == 0][:200]
    ood = {"uniform_gray": np.full((200, size, size, 3), 128, np.uint8),
           "gaussian_noise": rng.normal(128, 50, (200, size, size, 1)).clip(0, 255).astype(np.uint8).repeat(3, -1),
           "inverted_xray": 255 - negatives, "rotated_90_xray": np.rot90(negatives, axes=(1, 2)).copy()}
    report["out_of_domain"] = {}
    for name, batch in ood.items():
        s = scores(model, batch, device)
        report["out_of_domain"][name] = {"images": len(batch), "median_score": float(np.median(s)),
                                         "positive_rate": float((s >= threshold).mean())}

    # 5. Kermany pediátrico (test): ¿la adaptación olvidó el dominio de origen?
    kermany = pd.read_csv(args.kermany_partition)
    kermany = kermany[kermany.split == "test"].reset_index(drop=True)
    k_scores = scores(model, cached_images(kermany, args.kermany_images), device)
    report["kermany_pediatric_test"] = auc_and_rate(kermany.label.to_numpy(), k_scores, threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    occ.to_csv(args.out.parent / "occlusion_bbox_per_image.csv", index=False)
    write_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
