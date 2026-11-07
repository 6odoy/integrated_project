"""Análisis de fallas, calibración y valor predictivo del modelo final en NIH.

* Subgrupos (vista AP/PA, sexo, edad, neumonía sola o con co-hallazgos):
  sensibilidad y especificidad con IC bootstrap por paciente.
* Calibración: ECE (10 bins), Brier y diagrama de fiabilidad del modelo
  Kermany sin adaptar y del final. Se ajusta temperature scaling solo en
  adapt_val; la BCE ponderada desplaza las probabilidades hacia el positivo.
* Valor predictivo a prevalencias distintas de la artificial del 25 %.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from nih_common import patient_bootstrap  # noqa: F401  (fija sys.path de Fase 2)
from common import write_json

BINS = 10


def ece(y, p, bins=BINS):
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        inside = index == b
        if inside.any():
            total += inside.mean() * abs(y[inside].mean() - p[inside].mean())
    return float(total)


def reliability(y, p, bins=BINS):
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return [(p[index == b].mean(), y[index == b].mean(), int((index == b).sum())) for b in range(bins) if (index == b).any()]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_temperature(y, p):
    z = logit(p)
    nll = lambda t: -np.mean(y * -np.logaddexp(0, -z / t) + (1 - y) * -np.logaddexp(0, z / t))
    return float(minimize_scalar(nll, bounds=(.05, 20), method="bounded").x)


def subgroup_table(frame, threshold, n_boot):
    groups = {
        "Todos": np.ones(len(frame), bool),
        "Vista AP": frame.view.eq("AP").to_numpy(), "Vista PA": frame.view.eq("PA").to_numpy(),
        "Mujer": frame.sex.eq("F").to_numpy(), "Hombre": frame.sex.eq("M").to_numpy(),
        "Edad 18-39": (frame.age < 40).to_numpy(), "Edad 40-59": frame.age.between(40, 59).to_numpy(),
        "Edad >=60": (frame.age >= 60).to_numpy(),
    }
    rows = []
    for name, mask in groups.items():
        sub = frame[mask]
        y, s = sub.label.to_numpy(), sub.score.to_numpy()
        boot = patient_bootstrap(y, s, sub.patient_id.to_numpy(), threshold, n_boot, 42) if 0 < y.mean() < 1 else {}
        rows.append({"subgroup": name, "images": len(sub), "positives": int(y.sum()),
                     "sensitivity": float((s[y == 1] >= threshold).mean()) if (y == 1).any() else np.nan,
                     "sensitivity_ci": boot.get("sensitivity"),
                     "specificity": float((s[y == 0] < threshold).mean()) if (y == 0).any() else np.nan,
                     "specificity_ci": boot.get("specificity"), "auc_roc_ci": boot.get("auc_roc")})
    positives = frame[frame.label == 1]
    for name, mask in (("Neumonía sola", positives.pneumonia_only), ("Neumonía + co-hallazgos", ~positives.pneumonia_only)):
        s = positives.loc[mask, "score"].to_numpy()
        rows.append({"subgroup": name, "images": len(s), "positives": len(s),
                     "sensitivity": float((s >= threshold).mean()), "sensitivity_ci": None,
                     "specificity": np.nan, "specificity_ci": None, "auc_roc_ci": None})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final-run", required=True, type=Path)
    ap.add_argument("--zero-shot-run", required=True, type=Path, help="Salida de evaluate_zero_shot.py, misma semilla")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--figures", required=True, type=Path)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    final = pd.read_csv(args.final_run / "predictions_nih_external_test.csv")
    threshold = json.loads((args.final_run / "metrics.json").read_text())["threshold"]
    val = pd.read_csv(args.final_run / "predictions_nih_adapt_val.csv")
    zero = pd.read_csv(args.zero_shot_run / "predictions_nih_external_test_kermany_threshold.csv")
    args.out.mkdir(parents=True, exist_ok=True)

    table = subgroup_table(final, threshold, args.n_boot)
    table.to_csv(args.out / "subgroups.csv", index=False)

    # Errores del modelo final: qué tienen en común.
    errors = final.assign(error=np.where(final.label == 1, np.where(final.prediction == 0, "FN", "VP"),
                                         np.where(final.prediction == 1, "FP", "VN")))
    errors.groupby("error").agg(images=("study_id", "size"), ap_fraction=("view", lambda v: float((v == "AP").mean())),
                                age_median=("age", "median"), female_fraction=("sex", lambda v: float((v == "F").mean())),
                                pneumonia_only=("pneumonia_only", "mean"), score_median=("score", "median")).to_csv(
        args.out / "error_profile.csv")
    fn = final[(final.label == 1) & (final.prediction == 0)]
    co = fn.finding_labels.str.split("|").explode()
    co[co != "Pneumonia"].value_counts().rename("fn_count").to_frame().join(
        final[final.label == 1].finding_labels.str.split("|").explode().value_counts().rename("positive_count")
    ).assign(fn_rate=lambda d: d.fn_count / d.positive_count).to_csv(args.out / "false_negative_cofindings.csv")

    # Calibración.
    y = final.label.to_numpy()
    temperature = fit_temperature(val.label.to_numpy(), val.score.to_numpy())
    scaled = 1 / (1 + np.exp(-logit(final.score.to_numpy()) / temperature))
    calibration = {}
    curves = {"Kermany sin adaptar": (zero.label.to_numpy(), zero.score.to_numpy()),
              "Final (adaptado)": (y, final.score.to_numpy()),
              f"Final + temperatura (T={temperature:.2f})": (y, scaled)}
    for name, (yy, pp) in curves.items():
        calibration[name] = {"ece": ece(yy, pp), "brier": float(np.mean((pp - yy) ** 2)),
                             "mean_predicted": float(pp.mean()), "observed_prevalence": float(yy.mean())}
    # El umbral es monótono en la temperatura: la decisión no cambia, solo la lectura de la probabilidad.
    calibration["temperature_fitted_on"] = "NIH adapt_val"
    calibration["temperature"] = temperature

    # Valor predictivo según prevalencia (sensibilidad y especificidad fijas).
    sens = float((final.score[y == 1] >= threshold).mean())
    spec = float((final.score[y == 0] < threshold).mean())
    calibration["predictive_values"] = [
        {"prevalence": prev, "ppv": sens * prev / (sens * prev + (1 - spec) * (1 - prev)),
         "npv": spec * (1 - prev) / (spec * (1 - prev) + (1 - sens) * prev)}
        for prev in (.02, .05, .10, .25)]
    write_json(args.out / "calibration.json", calibration)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot([0, 1], [0, 1], color="0.6", linestyle=":", label="Calibración perfecta")
    for name, (yy, pp) in curves.items():
        points = np.array([(a, b) for a, b, _ in reliability(yy, pp)])
        axes[0].plot(points[:, 0], points[:, 1], marker="o", markersize=3,
                     label=f"{name} (ECE={calibration[name]['ece']:.3f})")
    axes[0].set(xlabel="Probabilidad predicha", ylabel="Frecuencia observada", title="Diagrama de fiabilidad (external_test)")
    axes[0].legend(fontsize=7)
    for label, color in ((0, "tab:blue"), (1, "tab:red")):
        axes[1].hist(final.score[y == label], bins=30, alpha=.55, color=color,
                     label="No Finding" if label == 0 else "Neumonía", density=True)
    axes[1].axvline(threshold, color="k", linestyle="--", linewidth=1, label=f"Umbral (adapt_val) = {threshold:.3f}")
    axes[1].set(xlabel="Puntaje del modelo final", ylabel="Densidad", title="Separación de clases")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    args.figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figures / "calibration.png", dpi=170)
    plt.close(fig)
    print(table.to_string(index=False))
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()
