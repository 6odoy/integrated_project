"""Grad-CAM del modelo final y de su origen Kermany, medido contra las cajas del radiólogo.

Para cada imagen de external_test con caja de neumonía se calcula:

* ``energy_in_box``: fracción de la masa del mapa dentro de la caja.
* ``box_area_fraction``: área de la caja sobre la imagen; es lo que obtendría
  un mapa uniforme, así que ``energy_in_box / box_area_fraction > 1`` indica
  concentración sobre la lesión.
* ``pointing_hit``: si el máximo del mapa cae dentro de la caja (pointing game).

En todo external_test se mide además la masa en el marco periférico (10 % de
cada borde), donde están los marcadores de lateralidad y el texto: es el
indicador de atajo que se contrasta con su área.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from audit_utils import cached_images, get_device, gradcam, load_model, normalized, scores
from nih_common import load_partition

NIH_SIZE = 1024
BORDER = .10


def border_mask(size: int) -> np.ndarray:
    mask = np.ones((size, size), dtype=bool)
    m = int(round(BORDER * size))
    mask[m:size - m, m:size - m] = False
    return mask


def box_mask(box, size: int) -> np.ndarray:
    x, y, w, h = (np.asarray(box) * size / NIH_SIZE)
    mask = np.zeros((size, size), dtype=bool)
    mask[int(np.floor(y)):int(np.ceil(y + h)), int(np.floor(x)):int(np.ceil(x + w))] = True
    return mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partition", required=True, type=Path)
    ap.add_argument("--images-root", required=True, type=Path)
    ap.add_argument("--bbox", required=True, type=Path)
    ap.add_argument("--final-run", required=True, type=Path)
    ap.add_argument("--source-run", required=True, type=Path, help="Modelo Kermany sin adaptar (misma semilla)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--figures", required=True, type=Path)
    args = ap.parse_args()
    device = get_device()
    test = load_partition(args.partition)["external_test"]
    images = cached_images(test, args.images_root)
    size = images.shape[1]
    bbox = pd.read_csv(args.bbox)
    bbox = bbox[bbox["Finding Label"] == "Pneumonia"].rename(columns={
        "Bbox [x": "x", "y": "y", "w": "w", "h]": "h"}).set_index("Image Index")
    frame = border_mask(size)
    rows, border_rows, maps = [], [], {}
    for name, run_dir in (("final", args.final_run), ("kermany_source", args.source_run)):
        model, threshold = load_model(run_dir, device)
        model_scores = scores(model, images, device)
        cams = gradcam(model, images, device)
        maps[name] = (cams, model_scores, threshold)
        for i, row in test.iterrows():
            mass = normalized(cams[i])
            border_rows.append({"model": name, "study_id": row.study_id, "label": row.label, "view": row.view,
                                "score": model_scores[i], "prediction": int(model_scores[i] >= threshold),
                                "border_energy": float(mass[frame].sum()), "border_area_fraction": float(frame.mean())})
            if row.image_path in bbox.index:
                boxes = bbox.loc[[row.image_path], ["x", "y", "w", "h"]].to_numpy()
                inside = np.zeros((size, size), dtype=bool)
                for box in boxes:
                    inside |= box_mask(box, size)
                peak = np.unravel_index(np.argmax(cams[i]), cams[i].shape)
                rows.append({"model": name, "study_id": row.study_id, "view": row.view, "score": model_scores[i],
                             "detected": int(model_scores[i] >= threshold), "energy_in_box": float(mass[inside].sum()),
                             "box_area_fraction": float(inside.mean()), "pointing_hit": bool(inside[peak])})
    args.out.mkdir(parents=True, exist_ok=True)
    per_image = pd.DataFrame(rows)
    per_image["concentration_ratio"] = per_image.energy_in_box / per_image.box_area_fraction
    per_image.to_csv(args.out / "gradcam_bbox_per_image.csv", index=False)
    border = pd.DataFrame(border_rows)
    border.to_csv(args.out / "gradcam_border_per_image.csv", index=False)
    summary = per_image.groupby("model").agg(
        images=("study_id", "size"), detected=("detected", "mean"), energy_in_box=("energy_in_box", "median"),
        box_area_fraction=("box_area_fraction", "median"), concentration_ratio=("concentration_ratio", "median"),
        pointing_game=("pointing_hit", "mean"), random_pointing_expected=("box_area_fraction", "mean"))
    summary = summary.join(border.groupby("model").agg(border_energy=("border_energy", "median"),
                                                       border_area_fraction=("border_area_fraction", "first")))
    summary.to_csv(args.out / "gradcam_summary.csv")
    print(summary.to_string())

    # Figura: casos con caja (antes y después del traslado) y errores del modelo final.
    cams, final_scores, threshold = maps["final"]
    source_cams = maps["kermany_source"][0]
    with_box = [i for i, p in enumerate(test.image_path) if p in bbox.index]
    rng = np.random.default_rng(42)
    picks = list(rng.choice(with_box, size=min(4, len(with_box)), replace=False))
    labels, predictions = test.label.to_numpy(), (final_scores >= threshold).astype(int)
    for wanted_label, wanted_pred in ((1, 0), (0, 1)):
        candidates = np.setdiff1d(np.flatnonzero((labels == wanted_label) & (predictions == wanted_pred)), picks)
        if len(candidates):
            # El error más confiado: es el que más interesa leer.
            order = np.argsort(final_scores[candidates])
            picks.append(int(candidates[order[0] if wanted_label == 1 else order[-1]]))
    fig, axes = plt.subplots(len(picks), 3, figsize=(7.2, 2.45 * len(picks)))
    for r, i in enumerate(picks):
        row = test.iloc[i]
        tag = {(1, 1): "VP", (1, 0): "FN", (0, 1): "FP", (0, 0): "VN"}[(int(labels[i]), int(predictions[i]))]
        for c, (title, cam) in enumerate((("Imagen", None), ("Kermany sin adaptar", source_cams[i]),
                                          ("Final adaptado", cams[i]))):
            ax = axes[r, c]
            ax.imshow(images[i][..., 0], cmap="gray")
            if cam is not None:
                ax.imshow(cam / (cam.max() + 1e-8), cmap="jet", alpha=.4)
            if row.image_path in bbox.index:
                for x, y, w, h in bbox.loc[[row.image_path], ["x", "y", "w", "h"]].to_numpy() * size / NIH_SIZE:
                    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="white", linewidth=1.2))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{tag} {row.view}\np={final_scores[i]:.2f}", fontsize=8)
    fig.suptitle("Grad-CAM en NIH external_test (caja blanca: radiólogo)", fontsize=10)
    fig.tight_layout()
    args.figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figures / "gradcam_examples.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
