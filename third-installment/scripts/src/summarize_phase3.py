"""Tablas LaTeX y figuras de la entrega final, exclusivamente desde artefactos guardados."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ("sensitivity", "specificity", "auc_roc", "auprc", "f1")
NAMES = {
    "baseline_random_forest": "Random Forest (histograma)",
    "zero_shot_kermany_threshold": "Kermany directo, umbral Kermany",
    "zero_shot_nih_val_threshold": "Kermany directo, umbral NIH val",
    "adapt_frozen": "Traslado: extractor congelado",
    "adapt_finetune_partial": "Traslado: fine-tuning parcial",
    "adapt_imagenet_finetune_partial": "Control: ImageNet $\\to$ NIH",
}


def comma(value, digits=3):
    return "--" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}f}".replace(".", "{,}")


def load(path: Path):
    return json.loads(path.read_text())


def collect(results: Path) -> pd.DataFrame:
    rows = []
    rf = results / "baseline_random_forest" / "metrics.json"
    if rf.exists():
        rows.append({"configuration": "baseline_random_forest", "seed": 42, **{m: load(rf)[m] for m in METRICS},
                     "constraint": load(rf)["specificity_constraint_satisfied"]})
    for path in sorted(results.glob("zero_shot/finetune_partial/seed*/metrics.json")):
        data = load(path)
        for key in ("kermany_threshold", "nih_val_threshold"):
            rows.append({"configuration": f"zero_shot_{key}", "seed": data["seed"], **{m: data[key][m] for m in METRICS},
                         "constraint": data[key]["specificity_constraint_satisfied"]})
    for path in sorted(results.glob("adapt/*/seed*/metrics.json")):
        data = load(path)
        rows.append({"configuration": f"adapt_{path.parent.parent.name}", "seed": data["seed"],
                     **{m: data[m] for m in METRICS}, "constraint": data["specificity_constraint_satisfied"],
                     "best_epoch": data["best_epoch"], "epochs_run": data["epochs_run"],
                     "elapsed_seconds": data["elapsed_seconds"], "trainable_parameters": data["trainable_parameters"],
                     "inference_ms": data["inference_ms_image"], "inference_p25": data["inference_ms_p25"],
                     "inference_p75": data["inference_ms_p75"], "best_val_auc": data["best_val_auc"]})
    return pd.DataFrame(rows)


def results_table(runs: pd.DataFrame, out: Path) -> pd.DataFrame:
    order = [k for k in NAMES if k in set(runs.configuration)]
    summary = runs.groupby("configuration").agg(
        **{m: (m, "mean") for m in METRICS}, **{f"{m}_sd": (m, lambda v: v.std(ddof=0)) for m in METRICS},
        seeds=("seed", "size"), constraint_runs=("constraint", "sum")).reindex(order)
    summary.to_csv(out / "nih_summary.csv")
    lines = ["\\begin{tabular}{lcccccc}", "\\toprule",
             "Configuración & Sens. & Espec. & AUC-ROC & AUPRC & F1 & Espec.$\\geq$0,80 \\\\", "\\midrule"]
    for key, r in summary.iterrows():
        cells = [comma(r[m]) + (f" $\\pm$ {comma(r[m + '_sd'])}" if r.seeds > 1 and m in ("sensitivity", "specificity", "auc_roc") else "")
                 for m in METRICS]
        lines.append(f"{NAMES[key]} & " + " & ".join(cells) + f" & {int(r.constraint_runs)}/{int(r.seeds)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "table_nih.tex").write_text("\n".join(lines) + "\n")
    return summary


def paired_table(results: Path, out: Path) -> None:
    labels = {"finetune_partial": "Parcial $-$ RF", "frozen": "Congelado $-$ RF",
              "partial_vs_frozen": "Parcial $-$ congelado", "kermany_vs_imagenet": "Kermany $\\to$ NIH $-$ ImageNet $\\to$ NIH"}
    rows = []
    for path in sorted(results.glob("paired/*.json")):
        name = path.stem
        kind = next(k for k in ("partial_vs_frozen", "kermany_vs_imagenet", "finetune_partial", "frozen") if name.startswith(k))
        seed = int(name.split("seed")[1].split("_")[0])
        diff = load(path)["difference_a_minus_b_bootstrap_95"]["metrics"]
        rows.append({"comparison": kind, "seed": seed,
                     **{f"{m}_{part}": (diff[m]["estimate"] if part == "est" else diff[m]["ci_95"][0 if part == "lo" else 1])
                        for m in ("sensitivity", "specificity", "auc_roc") for part in ("est", "lo", "hi")}})
    if not rows:
        return
    frame = pd.DataFrame(rows).sort_values(["comparison", "seed"])
    frame.to_csv(out / "nih_paired.csv", index=False)
    lines = ["\\begin{tabular}{llccc}", "\\toprule", "Contraste (A $-$ B) & Semilla & $\\Delta$ Sens. & $\\Delta$ Espec. & $\\Delta$ AUC \\\\", "\\midrule"]
    for kind in labels:
        for _, r in frame[frame.comparison == kind].iterrows():
            cells = [f"{comma(100 * r[f'{m}_est'], 1)} [{comma(100 * r[f'{m}_lo'], 1)}; {comma(100 * r[f'{m}_hi'], 1)}]"
                     for m in ("sensitivity", "specificity", "auc_roc")]
            lines.append(f"{labels[kind]} & {r.seed} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "table_paired.tex").write_text("\n".join(lines) + "\n")


def cost_table(runs: pd.DataFrame, kermany: Path, out: Path) -> None:
    rows = []
    for mode in ("frozen", "finetune_partial"):
        data = [load(p) for p in sorted((kermany / mode).glob("seed*/metrics.json"))]
        if data:
            rows.append({"stage": "Kermany", "mode": mode, "trainable_parameters": data[0]["trainable_parameters"],
                         "epochs": np.mean([d["epochs_run"] for d in data]),
                         "train_minutes": np.mean([d["elapsed_seconds"] for d in data]) / 60,
                         "inference_ms": np.median([d["inference_ms_image"] for d in data]),
                         "inference_p25": np.median([d["inference_ms_p25"] for d in data]),
                         "inference_p75": np.median([d["inference_ms_p75"] for d in data])})
    adapt = runs[runs.configuration.str.startswith("adapt_")]
    for key, group in adapt.groupby("configuration"):
        rows.append({"stage": "NIH", "mode": key.removeprefix("adapt_"), "trainable_parameters": group.trainable_parameters.iloc[0],
                     "epochs": group.epochs_run.mean(), "train_minutes": group.elapsed_seconds.mean() / 60,
                     "inference_ms": group.inference_ms.median(), "inference_p25": group.inference_p25.median(),
                     "inference_p75": group.inference_p75.median()})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "cost.csv", index=False)
    mode_names = {"frozen": "Congelado", "finetune_partial": "Fine-tuning parcial", "imagenet_finetune_partial": "ImageNet, parcial"}
    lines = ["\\begin{tabular}{llrrrc}", "\\toprule",
             "Etapa & Configuración & Parám. entrenables & Épocas & Min. entren. & Latencia ms [IQR] \\\\", "\\midrule"]
    for _, r in frame.iterrows():
        parameters = f"{int(r.trainable_parameters):,}".replace(",", ".")
        lines.append(f"{r.stage} & {mode_names[r['mode']]} & {parameters}"
                     f" & {comma(r.epochs, 1)} & {comma(r.train_minutes, 1)} & {comma(r.inference_ms, 1)} "
                     f"[{comma(r.inference_p25, 1)}; {comma(r.inference_p75, 1)}] \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "table_cost.tex").write_text("\n".join(lines) + "\n")


def learning_curve(results: Path, runs: pd.DataFrame, figures: Path, out: Path) -> None:
    rows = []
    for path in results.glob("curve/fraction*/seed*/metrics.json"):
        data = load(path)
        rows.append({"fraction": data["train_fraction"], "seed": data["seed"], "images": data["adapt_train_images"],
                     "auc_roc": data["auc_roc"], "sensitivity": data["sensitivity"], "specificity": data["specificity"]})
    full = runs[runs.configuration == "adapt_finetune_partial"]
    rows += [{"fraction": 1.0, "seed": r.seed, "images": np.nan, "auc_roc": r.auc_roc, "sensitivity": r.sensitivity,
              "specificity": r.specificity} for _, r in full.iterrows()]
    if not rows:
        return
    curve = pd.DataFrame(rows).groupby("fraction").agg(
        images=("images", "mean"), auc=("auc_roc", "mean"), auc_sd=("auc_roc", lambda v: v.std(ddof=0)),
        sens=("sensitivity", "mean"), sens_sd=("sensitivity", lambda v: v.std(ddof=0)),
        spec=("specificity", "mean"), spec_sd=("specificity", lambda v: v.std(ddof=0)), seeds=("seed", "size"))
    curve.to_csv(out / "learning_curve.csv")
    reference = runs.groupby("configuration").auc_roc.mean()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = 100 * curve.index.to_numpy()
    for column, label, color in (("auc", "AUC-ROC", "tab:blue"), ("sens", "Sensibilidad", "tab:red"), ("spec", "Especificidad", "tab:green")):
        ax.errorbar(x, curve[column], yerr=curve[column + "_sd"], marker="o", capsize=3, label=label, color=color)
    for key, style in (("zero_shot_nih_val_threshold", ":"), ("baseline_random_forest", "--"), ("adapt_imagenet_finetune_partial", "-.")):
        if key in reference:
            ax.axhline(reference[key], color="0.4", linestyle=style, linewidth=1, label=f"AUC {NAMES[key]}")
    ax.set(xscale="log", xlabel="% de pacientes de adapt_train usados (escala log)", ylabel="Métrica en external_test",
           title="Traslado con datos escasos (fine-tuning parcial, media $\\pm$ DE de 3 semillas)")
    ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in x])
    ax.legend(fontsize=6.5, loc="lower right"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(figures / "learning_curve.png", dpi=170); plt.close(fig)


def training_curves(results: Path, figures: Path) -> None:
    names = {"frozen": "Congelado", "finetune_partial": "Parcial", "imagenet_finetune_partial": "ImageNet"}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    colors = {"frozen": "tab:blue", "finetune_partial": "tab:red", "imagenet_finetune_partial": "tab:green"}
    for path in sorted(results.glob("adapt/*/seed*/history.csv")):
        mode, seed = path.parent.parent.name, path.parent.name
        h = pd.read_csv(path); c = colors[mode]; label = f"{names[mode]} {seed}"
        axes[0].plot(h.epoch, h.train_loss, color=c, alpha=.8, label=label)
        axes[0].plot(h.epoch, h.val_loss, color=c, alpha=.8, linestyle="--")
        axes[1].plot(h.epoch, h.val_auc, color=c, alpha=.8, label=label)
        axes[2].plot(h.epoch, h.gradient_norm, color=c, alpha=.8, label=label)
    axes[0].set(title="BCE ponderada (continua: train; discontinua: val)", xlabel="Época", ylabel="Pérdida")
    axes[1].set(title="AUC en adapt_val", xlabel="Época", ylabel="AUC")
    axes[2].set(title="Norma media del gradiente", xlabel="Época", ylabel="Norma", yscale="log")
    for ax in axes: ax.grid(alpha=.25)
    axes[1].legend(fontsize=6, ncol=1)
    fig.tight_layout(); fig.savefig(figures / "nih_training_curves.png", dpi=170); plt.close(fig)


def audit_tables(results: Path, out: Path) -> None:
    audit = results / "audit"
    if (audit / "subgroups.csv").exists():
        sub = pd.read_csv(audit / "subgroups.csv")
        def ci(text):
            if not isinstance(text, str) or not text.startswith("["):
                return ""
            lo, hi = json.loads(text)
            return f" [{comma(lo, 2)}; {comma(hi, 2)}]"
        lines = ["\\begin{tabular}{lrrcc}", "\\toprule", "Subgrupo & Imágenes & Positivas & Sensibilidad [IC 95\\%] & Especificidad [IC 95\\%] \\\\", "\\midrule"]
        for _, r in sub.iterrows():
            lines.append(f"{r.subgroup} & {r.images} & {r.positives} & {comma(r.sensitivity, 2)}{ci(r.sensitivity_ci)} & "
                         f"{comma(r.specificity, 2)}{ci(r.specificity_ci)} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (out / "table_subgroups.tex").write_text("\n".join(lines) + "\n")
    if (audit / "known_response_tests.json").exists():
        t = load(audit / "known_response_tests.json")
        g = pd.read_csv(audit / "gradcam_summary.csv", index_col=0) if (audit / "gradcam_summary.csv").exists() else None
        rows = [("Original", t["original"]["auc_roc"], t["original"]["positive_rate"], "Referencia"),
                ("Tórax central ocluido (60\\%)", t["center_occluded"]["auc_roc"], t["center_occluded"]["positive_rate"], "AUC $\\to$ 0,5"),
                ("Marco periférico ocluido (10\\%)", t["border_occluded"]["auc_roc"], t["border_occluded"]["positive_rate"], "AUC $\\approx$ original"),
                ("Volteo horizontal", t["horizontal_flip"]["auc_roc"], t["horizontal_flip"]["positive_rate"], "AUC $\\approx$ original"),
                ("Kermany pediátrico (test)", t["kermany_pediatric_test"]["auc_roc"], t["kermany_pediatric_test"]["positive_rate"], "Retención del origen")]
        lines = ["\\begin{tabular}{lccl}", "\\toprule", "Prueba & AUC-ROC & Tasa de positivos & Respuesta esperada \\\\", "\\midrule"]
        lines += [f"{a} & {comma(b)} & {comma(c)} & {d} \\\\" for a, b, c, d in rows]
        lines.append("\\midrule")
        for name, label in (("uniform_gray", "Gris uniforme"), ("gaussian_noise", "Ruido gaussiano"),
                            ("inverted_xray", "Radiografía invertida"), ("rotated_90_xray", "Radiografía rotada 90°")):
            o = t["out_of_domain"][name]
            lines.append(f"{label} (fuera de dominio) & -- & {comma(o['positive_rate'])} & Sin positivos confiados \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (out / "table_tests.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--kermany-results", required=True, type=Path)
    ap.add_argument("--figures", required=True, type=Path)
    ap.add_argument("--generated", required=True, type=Path)
    args = ap.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True); args.generated.mkdir(parents=True, exist_ok=True)
    runs = collect(args.results)
    if runs.empty:
        raise SystemExit("No hay resultados de Fase 3")
    runs.to_csv(args.generated / "nih_all_runs.csv", index=False)
    print(results_table(runs, args.generated).round(3).to_string())
    paired_table(args.results, args.generated)
    cost_table(runs, args.kermany_results, args.generated)
    learning_curve(args.results, runs, args.figures, args.generated)
    training_curves(args.results, args.figures)
    audit_tables(args.results, args.generated)


if __name__ == "__main__":
    main()
