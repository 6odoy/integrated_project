.PHONY: help download-kermany phase2-prepare phase2-train phase2-figures phase2-baseline phase2-compare-rf-finetune

help:
	@echo "Objetivos disponibles:"
	@echo "  make download-kermany  Descarga y verifica el dataset Chest X-Ray Pneumonia (Kermany)."
	@echo "  make phase2-prepare   Crea y verifica la partición por paciente de Fase 2."
	@echo "  make phase2-train     Ejecuta 2 configuraciones x 3 semillas (CPU, puede tardar)."
	@echo "  make phase2-figures   Genera tablas y figuras que consume el informe LaTeX."
	@echo "  make phase2-baseline  Entrena Random Forest en la misma partición de Fase 2."
	@echo "  make phase2-compare-rf-finetune  IC bootstrap pareado RF vs. CNN (3 semillas)."

download-kermany:
	bash first-installment/scripts/download_kermany.sh

phase2-prepare:
	conda run --no-capture-output --name proyecto_deep python second-installment/scripts/src/prepare_kermany.py --manifest second-installment/scripts/data_manifest_kermany.csv --out second-installment/scripts/results/metadata

phase2-train: phase2-prepare
	TORCH_HOME=/tmp/phase2-torch conda run --no-capture-output --name proyecto_deep python second-installment/scripts/src/run_phase2.py --partition second-installment/scripts/results/metadata/partition_kermany.csv --images-root first-installment/scripts/data/chest_xray --config second-installment/scripts/configs/densenet121.json --results second-installment/scripts/results
	$(MAKE) phase2-figures

phase2-figures:
	MPLCONFIGDIR=/tmp/matplotlib-phase2 conda run --no-capture-output --name proyecto_deep python second-installment/scripts/src/figures.py --results second-installment/scripts/results --figures second-installment/report/figures --generated second-installment/report/generated

phase2-baseline: phase2-prepare
	conda run --no-capture-output --name proyecto_deep python second-installment/scripts/src/baseline_random_forest.py --partition second-installment/scripts/results/metadata/partition_kermany.csv --images-root first-installment/scripts/data/chest_xray --out second-installment/scripts/results/kermany/baseline_random_forest

phase2-compare-rf-finetune: phase2-baseline
	conda run --no-capture-output --name proyecto_deep python second-installment/scripts/src/compare_paired.py --model-a second-installment/scripts/results/kermany/finetune_partial/seed42/predictions_kermany_test.csv --metrics-a second-installment/scripts/results/kermany/finetune_partial/seed42/metrics.json --model-b second-installment/scripts/results/kermany/baseline_random_forest/predictions_kermany_test.csv --metrics-b second-installment/scripts/results/kermany/baseline_random_forest/metrics.json --out second-installment/scripts/results/kermany/paired_finetune_partial_seed42_vs_random_forest.json
	conda run --no-capture-output --name proyecto_deep python second-installment/scripts/src/compare_paired.py --model-a second-installment/scripts/results/kermany/finetune_partial/seed43/predictions_kermany_test.csv --metrics-a second-installment/scripts/results/kermany/finetune_partial/seed43/metrics.json --model-b second-installment/scripts/results/kermany/baseline_random_forest/predictions_kermany_test.csv --metrics-b second-installment/scripts/results/kermany/baseline_random_forest/metrics.json --out second-installment/scripts/results/kermany/paired_finetune_partial_seed43_vs_random_forest.json
	conda run --no-capture-output --name proyecto_deep python second-installment/scripts/src/compare_paired.py --model-a second-installment/scripts/results/kermany/finetune_partial/seed44/predictions_kermany_test.csv --metrics-a second-installment/scripts/results/kermany/finetune_partial/seed44/metrics.json --model-b second-installment/scripts/results/kermany/baseline_random_forest/predictions_kermany_test.csv --metrics-b second-installment/scripts/results/kermany/baseline_random_forest/metrics.json --out second-installment/scripts/results/kermany/paired_finetune_partial_seed44_vs_random_forest.json
