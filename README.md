# Integrated Project — Pneumonia Detection in Chest X-Rays

The Deep Learning project that critically replicates Rahman et al.'s (2020)
approach to pneumonia detection in chest X-rays. Beyond reproducing the method,
it evaluates whether a model trained on pediatric images from Guangzhou
(Kermany) retains its sensitivity when evaluated, without retraining, on an
adult Latin American population (BRAX).

The first installment containes the article review, local problem framing, and a
reproducible baseline. The full report is available in
[`first-installment/report/main.tex`](first-installment/report/main.tex).

## Research question and scope

> Does a pretrained CNN fine-tuned on Kermany retain sufficient sensitivity to
> support triage when evaluated on an adult Latin American population (BRAX), and
> does it outperform a classical baseline under an equivalent data split?

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
| Planned external evaluation | BRAX | Adult chest X-rays from Hospital Albert Einstein, São Paulo. Only frontal images with positive/negative labels are used; it does not participate in training or tuning. |
| Alternative if BRAX is unavailable | PadChest | Adult Spanish dataset for external evaluation. |

The external evaluation measures the domain gap between pediatric Guangzhou and
adult São Paulo data; it cannot establish generalizability to all of Colombia.
In BRAX, labels are extracted from radiology reports through NLP, and uncertain
cases (`-1`) are excluded from binary evaluation and reserved for Phase 3.

## Project status

- Completed (Phase 1): critical review of Rahman et al., local problem framing,
  methodological definition, and a classical baseline on Kermany.
- Next (Phase 2): fine-tuning DenseNet201—or DenseNet121 depending on the compute budget—at 224 × 224, with three seeds, augmentation only during training, weighted binary loss, and early stopping based on validation AUC.
- Next (Phase 2): internal Kermany testing and external BRAX evaluation without
  retraining or selecting hyperparameters on the external dataset.
- Next (Phase 3): Grad-CAM audit, uncertainty analysis, and assessment of
  shortcuts such as text, hospital markers, or regions outside the lung
  parenchyma.

The CNN will only be justified if it exceeds the best baseline by at least 5%
in sensitivity (non-overlapping 95% bootstrap confidence intervals), sustains
the advantage across all three seeds, has viable inference time, and produces
clinically plausible Grad-CAM maps.

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

## Structure (for now)

```text
.
├── Makefile
├── requirements.txt
└── first-installment/
    ├── report/
    │   ├── main.tex              # First-deliverable report
    │   ├── referencias.bib
    │   └── figures/              # Figures used by the report
    └── scripts/
        ├── download_kermany.sh   # Dataset download and validation
        ├── data/                 # Local dataset and manifests
        ├── results/kermany/      # Reference baseline results
        └── src/
            ├── manifest_kermany.py
            ├── data.py
            ├── metricas.py
            ├── baseline.py
            └── figuras.py
```

## References

- Rahman et al. (2020): replicated article and comparison of AlexNet, ResNet18,
  DenseNet201, and SqueezeNet.
- Kermany et al. (2018): pediatric dataset used for training and internal
  testing.
- Reis et al. (2022): BRAX, the adult Brazilian external dataset.

Full citations and methodological limitations are available in the
[report](first-installment/report/main.tex).
