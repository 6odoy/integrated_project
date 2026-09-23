# Integrated Project — Pneumonia Detection in Chest X-Rays

The Deep Learning project that critically replicates Rahman et al.'s (2020)
approach to pneumonia detection in chest X-rays. Beyond reproducing the method,
it evaluates both its direct domain shift and its patient-level local
adaptation from pediatric images from Guangzhou (Kermany) to an adult Latin
American population (BRAX).

The first installment contains the article review, local problem framing, and a
reproducible baseline: [`first-installment/report/main.tex`](first-installment/report/main.tex).
The second installment replicates the transfer approach with DenseNet121 and
tests it internally on Kermany:
[`second-installment/report/main.tex`](second-installment/report/main.tex).

## Research question and scope

> Does a pretrained CNN fine-tuned on Kermany, and then adapted with an
> isolated local BRAX partition, retain sufficient sensitivity for triage and
> outperform a classical baseline under an equivalent data split?

The task is binary image-level classification: `1 = pneumonia` and `0 = no
findings`. Pneumonia sensitivity is prioritized, subject to a minimum
specificity of 0.80; AUC-ROC, AUPRC, and F1 are also reported.

The scope deliberately excludes bacterial-versus-viral classification because
BRAX does not provide an equivalent label. This is experimental decision support
for triage, not a clinical system ready for diagnosis or deployment.

## Data

| Role | Dataset | Population and use |
|---|---|---|
| Training, validation, and internal testing | Chest X-Ray Pneumonia / Kermany | Pediatric chest X-rays from Guangzhou. A patient-level 70/15/15 split is rebuilt; Kaggle's original split is not used. |
| Local adaptation and isolated external test | BRAX | Adult chest X-rays from Hospital Albert Einstein, São Paulo. Only frontal binary cases are used; patient-level `adapt_train/adapt_val/external_test` prevents test leakage. |
| Alternative if BRAX is unavailable | PadChest | Adult Spanish dataset for external evaluation. |

The external evaluation measures the domain gap between pediatric Guangzhou and
adult São Paulo data; it cannot establish generalizability to all of Colombia.
In BRAX, labels are extracted from radiology reports through NLP, and uncertain
cases (`-1`) are excluded from binary evaluation and reserved for Phase 3.

## Project status

- Completed (Phase 1): critical review of Rahman et al., local problem framing,
  methodological definition, and a classical baseline on Kermany.
- Completed (Phase 2): transfer learning with DenseNet121 at 224 × 224 across
  seeds 42, 43 and 44, augmentation only during training, weighted binary loss,
  and early stopping on validation AUC. DenseNet201 was dropped for compute
  budget; the reduction is declared in the report.
- Completed (Phase 2): internal Kermany testing of both transfer
  configurations, six runs in 73.4 minutes.
- Ready to execute (Phase 2): BRAX adaptation and isolated external test.
  `prepare_brax.py` and `adapt_brax.py` implement the protocol, but BRAX is
  not available locally, so no local-transfer metric is reported.
- Next (Phase 3): Grad-CAM audit, uncertainty analysis, and assessment of
  shortcuts such as text, hospital markers, or regions outside the lung
  parenchyma.

### Phase 2 results on the Kermany test split

| Configuration | Sens. | Spec. | AUC-ROC | AUPRC | F1 |
|---|---|---|---|---|---|
| Frozen extractor | 0.959 ± 0.004 | 0.792 ± 0.008 | 0.978 ± 0.001 | 0.992 | 0.944 |
| Partial fine-tuning | **0.992 ± 0.001** | **0.869 ± 0.016** | **0.996 ± 0.001** | **0.999** | **0.973** |

The CNN will only be justified if it exceeds the best baseline by at least 5%
in sensitivity (non-overlapping 95% bootstrap confidence intervals), sustains
the advantage across all three seeds, has viable inference time, and produces
clinically plausible Grad-CAM maps.

Partial fine-tuning clears the four quantifiable requirements against the
Phase 1 Random Forest: +5.56 points of sensitivity, per-seed confidence bounds
(0.985, 0.987, 0.984) above the baseline's upper bound of 0.955, an advantage
in all three seeds, and 11.9 ms per image. The frozen extractor gains only
2.26 points and its intervals overlap the baseline. The Grad-CAM requirement
belongs to Phase 3, so the CNN is not yet justified — it only clears the
numerical bar.

Two caveats the report develops. All three frozen runs fall below the 0.80
specificity floor on test (0.781, 0.797, 0.797) because the threshold is picked
exactly on the constraint boundary and keeps no margin; their bootstrap
intervals still contain 0.80. And the uncertainty set reserved for Phase 3 is
empty: Kermany carries no `-1` labels, so that analysis depends entirely on
BRAX.

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
figures. It took 73.4 minutes on an Apple Silicon machine using PyTorch's MPS
backend; expect considerably longer on CPU only. Runs that already hold a
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

After obtaining BRAX legally, normalize its metadata into a pre-registered
patient-level adaptation split. `external_test` stays out of training,
early stopping, threshold selection, and model selection; `adapt_train` and
`adapt_val` are the local transfer data:

```bash
python second-installment/scripts/src/prepare_brax.py --metadata <csv> --out <out.csv>
python second-installment/scripts/src/adapt_brax.py --manifest <out.csv> \
  --images-root <brax-images> --source-run <kermany-run> \
  --config second-installment/scripts/configs/densenet121.json \
  --mode frozen --out <results-frozen>
python second-installment/scripts/src/adapt_brax.py --manifest <out.csv> \
  --images-root <brax-images> --source-run <kermany-run> \
  --config second-installment/scripts/configs/densenet121.json \
  --mode finetune_partial --out <results-partial>
```

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
```

## References

- Rahman et al. (2020): replicated article and comparison of AlexNet, ResNet18,
  DenseNet201, and SqueezeNet.
- Kermany et al. (2018): pediatric dataset used for training and internal
  testing.
- Reis et al. (2022): BRAX, the adult Brazilian external dataset.

Full citations and methodological limitations are available in the
[report](first-installment/report/main.tex).
