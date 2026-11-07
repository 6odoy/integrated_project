# Integrated Project — Pneumonia Detection in Chest X-Rays

The Deep Learning project that critically replicates Rahman et al.'s (2020)
transfer-learning approach to pneumonia detection in chest X-rays, transfers it
from pediatric images from Guangzhou (Kermany) to adults (NIH ChestX-ray14), and
audits what the transferred model actually learned.

| Installment | Report |
|---|---|
| Phase 1: article review, local problem, baseline | [`first-installment/report/main.tex`](first-installment/report/main.tex) |
| Phase 2: replication on Kermany and transfer strategy | [`second-installment/report/main.tex`](second-installment/report/main.tex) |
| Phase 3: transfer to NIH, explainability audit, verdict | [`third-installment/report/main.tex`](third-installment/report/main.tex) |

## Research question and scope

> Does a pretrained CNN, adapted with an isolated patient-level partition of
> adult chest X-rays, retain enough sensitivity for triage, outperform a
> classical baseline under the same split, and rely on clinically pertinent
> signal rather than shortcuts?

The task is binary image-level classification: `1 = pneumonia` and `0 = no
findings`. Pneumonia sensitivity is prioritized, subject to a minimum
specificity of 0.80 fixed on validation; AUC-ROC, AUPRC, and F1 are also
reported. This is experimental decision support for triage, not a clinical
system ready for diagnosis or deployment.

## Data

| Role | Dataset | Population and use |
|---|---|---|
| Replication | Chest X-Ray Pneumonia / Kermany | Pediatric chest X-rays from Guangzhou. A patient-level 70/15/15 split is rebuilt; Kaggle's original split is not used. |
| Local transfer and isolated external test | NIH ChestX-ray14 | Adult chest X-rays from the NIH Clinical Center. Case-control subset (prevalence 0.25), patient-level `adapt_train/adapt_val/external_test`; radiologist bounding boxes reserved for the Grad-CAM audit. |

BRAX (adult, São Paulo) was the planned local dataset but access was not
granted; the professor approved NIH as the replacement, keeping the BRAX
protocol unchanged. NIH is not Latin American, so the transfer measures the
pediatric → adult and Chinese → US shift and says nothing about Colombian
patients. NIH labels come from NLP over radiology reports.

## Project status

- Completed (Phase 1): critical review of Rahman et al., local problem framing,
  and a Random Forest baseline.
- Completed (Phase 2, re-run for Phase 3): DenseNet121 transfer on Kermany,
  frozen extractor vs. partial fine-tuning, seeds 42/43/44, with validation
  loss logged. Phase 3 fixed a NORMAL-patient grouping leak (3,427 → 3,288
  patients) and a BatchNorm layer that kept updating in the "frozen" extractor.
- Completed (Phase 3): zero-shot evaluation on NIH, adaptation from Kermany
  (both modes) and from ImageNet, a data-scarcity curve, Random Forest on the
  same split, paired patient bootstrap, Grad-CAM against radiologist boxes,
  known-response tests, subgroup and calibration analysis, and the replication
  gap with Rahman et al. Details in [`third-installment/README.md`](third-installment/README.md).

### Main results

Kermany test (mean ± SD over three seeds):

| Configuration | Sens. | Spec. | AUC-ROC |
|---|---|---|---|
| Frozen extractor | 0.969 ± 0.003 | 0.811 ± 0.012 | 0.982 |
| Partial fine-tuning | 0.991 ± 0.001 | 0.852 ± 0.023 | 0.994 |

NIH `external_test` (1,484 images):

| Configuration | Sens. | Spec. | AUC-ROC |
|---|---|---|---|
| Random Forest | 0.639 | 0.820 | 0.804 |
| Kermany model, no adaptation, threshold re-picked on NIH | 0.723 | 0.817 | 0.835 |
| Kermany → NIH, frozen | 0.776 ± 0.053 | 0.816 ± 0.007 | 0.865 ± 0.024 |
| Kermany → NIH, partial | 0.818 ± 0.003 | 0.816 ± 0.014 | 0.890 ± 0.005 |
| ImageNet → NIH, partial (final model) | 0.845 ± 0.008 | 0.806 ± 0.002 | 0.894 ± 0.002 |

The CNN beats the Random Forest in every seed (paired 95% intervals exclude
zero), but the Kermany step adds nothing over ImageNet. The audit shows the
model uses the lesion (occluding the radiologist box lowers the score far more
than the mirrored region) yet also leans on the AP/portable acquisition view:
specificity is 0.43 on AP films and sensitivity 0.62 on PA films. Probabilities
are shifted by the class weight and are corrected by Platt scaling (ECE 0.165 →
0.042); the model has no out-of-distribution rejection.

## Reproduce the current baseline

The environment was tested with Python 3.13. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Conda can also be used:

```bash
conda create -n proyecto_deep python=3.13 pip
conda activate proyecto_deep
pip install -r requirements.txt
```

Download, validate with SHA-256, and unpack Kermany:

```bash
make download-kermany
```

Build the manifest, run the baseline, and generate the report figures:

```bash
cd first-installment/scripts
python src/manifest_kermany.py --root data/chest_xray --out data/manifest_kermany.csv
python src/baseline.py --manifest data/manifest_kermany.csv \
  --images-root data/chest_xray --out results/kermany
python src/figuras.py
```

The last step writes figures to `first-installment/report/figures/`, the path
used by the report. To shorten a local trial run, `baseline.py` accepts
`--n-boot`; its default value for reportable results is `1000`.

## Reproduce Phase 2

From the repository root, with `make download-kermany` already run:

```bash
make phase2-prepare
make phase2-train
make phase2-figures
```

`phase2-train` runs two configurations across three seeds and chains the
figures. The original Phase 2 sweep took 73.4 minutes on Apple Silicon (MPS);
the Phase 3 re-run took about 1 h 50 min on an NVIDIA GTX 1650, because the
frozen runs, with BatchNorm now truly frozen, reach the 30-epoch cap. Expect
considerably longer on CPU only. Runs that already hold a
`metrics.json` are skipped, so an interrupted sweep resumes where it stopped.

Each run writes to `second-installment/scripts/results/kermany/<mode>/seed<n>/`:
`history.csv`, `metrics.json`, `run_config.json`, `predictions_kermany_test.csv`
and `best_checkpoint.pt`. The checkpoints are 27 MB each and stay untracked;
everything else is versioned as the evidence the report consumes.

The dataset is decoded and resized once into RAM as uint8 (about 0.82 GB for
the three splits) instead of being re-read every epoch. This is what makes the
run practical on machines whose endpoint security software inspects every file
open. It prepends the same resize the transform used to apply, so results are
numerically equivalent.

The BRAX scripts (`prepare_brax.py`, `adapt_brax.py`) are kept for when access
is granted; the local transfer actually executed is the NIH one below.

## Reproduce Phase 3

With Kermany downloaded, everything else runs from one command:

```bash
make final    # Phase 2 re-run, NIH data, transfer, audit, tables, replication gap
make resume   # the same, in the background under a supervisor that restarts stalls
make status   # finished stages, last log lines and GPU usage
```

Every stage is resumable. The NIH-specific steps, data sources and outputs are
described in [`third-installment/README.md`](third-installment/README.md).

## Reproducible baseline design

The pipeline accepts a CSV manifest with one row per image:

```text
patient_id,study_id,image_path,view,label
```

- Retains only frontal `PA` and `AP` views.
- Treats `label = 1` as pneumonia and `label = 0` as no findings.
- Separates `label = -1` as the uncertainty set for Phase 3.
- Uses a stratified 70/15/15 patient-level split with seed 42, and verifies that
  no patient appears in more than one split.
- Selects hyperparameters and the decision threshold exclusively on validation
  data. The threshold maximizes sensitivity subject to specificity of at least
  0.80.
- Computes 95% confidence intervals with 1,000 patient-level bootstrap samples.

Four references are compared under the same split: a majority-class predictor,
logistic regression with PCA, Random Forest, and an MLP with one 64-unit hidden
layer. Features are grayscale intensity histograms after resizing to 128 × 128;
they do not replace the planned CNN, but establish a verifiable minimum.

## Results and artifacts

The run writes its outputs to `first-installment/scripts/results/kermany/`:

- `particion.csv`: assignment of each image to training, validation, or test.
- `conjunto_incertidumbre_fase3.csv`: observations with uncertain labels.
- `cuadro3.csv` and `cuadro3.tex`: metrics and confidence intervals.
- `config.json`: seed, specificity constraint, and selected hyperparameters.

Included reference results correspond to Phase 1. Downloaded data, images,
manifests, and generated splits are local artifacts and must not be versioned.

`first-installment/scripts/data/rahman_tabla4.csv` currently contains
documented DenseNet201 values. The AlexNet, ResNet18, and SqueezeNet figures
must be manually transcribed from Table 4 of Rahman et al.; the script does not
infer them.

## Structure

```text
.
├── Makefile
├── requirements.txt
├── first-installment/
│   ├── report/
│   │   ├── main.tex              # First-deliverable report
│   │   ├── referencias.bib
│   │   └── figures/              # Figures used by the report
│   └── scripts/
│       ├── download_kermany.sh   # Dataset download and validation
│       ├── data/                 # Local dataset and manifests
│       ├── results/kermany/      # Reference baseline results
│       └── src/
│           ├── manifest_kermany.py
│           ├── data.py
│           ├── metricas.py
│           ├── baseline.py
│           └── figuras.py
└── second-installment/
    ├── report/
    │   ├── main.tex              # Second-deliverable report
    │   ├── referencias.bib
    │   ├── figures/              # training_curves.png
    │   └── generated/            # Tables and CSVs built from metrics.json
    └── scripts/
        ├── data_manifest_kermany.csv
        ├── configs/densenet121.json
        ├── results/
        │   ├── metadata/         # Patient-level partition and summary
        │   └── kermany/          # Six runs; checkpoints untracked
        └── src/
            ├── common.py         # Split, metrics, threshold, bootstrap
            ├── prepare_kermany.py
            ├── train.py          # One transfer run, freezes the threshold
            ├── run_phase2.py     # Orchestrates 2 configurations x 3 seeds
            ├── prepare_brax.py
            ├── evaluate_brax.py
            └── figures.py
└── third-installment/
    ├── README.md                 # NIH data, experiments and unattended runs
    ├── report/
    │   ├── main.tex              # Final report
    │   ├── figures/              # Grad-CAM, calibration, curves
    │   └── generated/            # Tables built from the results
    └── scripts/
        ├── download_nih.sh       # NIH metadata with SHA-256
        ├── run_final.sh          # Stage runner behind `make resume`
        ├── supervise_final.sh    # Restarts the runner if it dies or stalls
        ├── results/              # NIH partition, runs, audit; checkpoints untracked
        └── src/                  # prepare, fetch, adapt, audit and summary scripts
```

## References

- Rahman et al. (2020): replicated article and comparison of AlexNet, ResNet18,
  DenseNet201, and SqueezeNet.
- Kermany et al. (2018): pediatric dataset used for training and internal
  testing.
- Wang et al. (2017): NIH ChestX-ray14, the adult dataset used for the local
  transfer.
- Reis et al. (2022): BRAX, the adult Brazilian dataset originally planned.

Full citations and methodological limitations are available in the
[report](first-installment/report/main.tex).
