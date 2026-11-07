# Intérprete del proyecto: el venv local si existe; se puede sobrescribir con
# `make PYTHON="conda run --no-capture-output --name proyecto_deep python" ...`.
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: help download-kermany phase2-prepare phase2-train phase2-figures phase2-baseline phase2-compare-rf-finetune

help:
	@echo "Objetivos disponibles:"
	@echo "  make download-kermany  Descarga y verifica el dataset Chest X-Ray Pneumonia (Kermany)."
	@echo "  make phase2-prepare   Crea y verifica la partición por paciente de Fase 2."
	@echo "  make phase2-train     Ejecuta 2 configuraciones x 3 semillas (CPU, puede tardar)."
	@echo "  make phase2-figures   Genera tablas y figuras que consume el informe LaTeX."
	@echo "  make phase2-baseline  Entrena Random Forest en la misma partición de Fase 2."
	@echo "  make phase2-compare-rf-finetune  IC bootstrap pareado RF vs. CNN (3 semillas)."
	@echo "  make phase3-data      Metadatos NIH, partición por paciente y descarga selectiva de imágenes."
	@echo "  make phase3-run       Evaluación directa, traslado (2 modos + control ImageNet), curva y RF en NIH."
	@echo "  make phase3-audit     Elige el modelo final por validación; Grad-CAM, pruebas y fallas."
	@echo "  make phase3-report    Tablas y figuras de la entrega final."
	@echo "  make final            Todo lo anterior en orden; reanudable tras un corte."
	@echo "  make resume           Igual que final, en segundo plano y supervisado (reinicia si se atasca)."
	@echo "  make status           Etapas terminadas, estado del supervisor y uso de GPU."

download-kermany:
	bash first-installment/scripts/download_kermany.sh

phase2-prepare:
	$(PYTHON) second-installment/scripts/src/prepare_kermany.py --manifest second-installment/scripts/data_manifest_kermany.csv --out second-installment/scripts/results/metadata

phase2-train: phase2-prepare
	TORCH_HOME=/tmp/phase2-torch $(PYTHON) second-installment/scripts/src/run_phase2.py --partition second-installment/scripts/results/metadata/partition_kermany.csv --images-root first-installment/scripts/data/chest_xray --config second-installment/scripts/configs/densenet121.json --results second-installment/scripts/results
	$(MAKE) phase2-figures

phase2-figures:
	MPLCONFIGDIR=/tmp/matplotlib-phase2 $(PYTHON) second-installment/scripts/src/figures.py --results second-installment/scripts/results --figures second-installment/report/figures --generated second-installment/report/generated

phase2-baseline: phase2-prepare
	$(PYTHON) second-installment/scripts/src/baseline_random_forest.py --partition second-installment/scripts/results/metadata/partition_kermany.csv --images-root first-installment/scripts/data/chest_xray --out second-installment/scripts/results/kermany/baseline_random_forest

phase2-compare-rf-finetune: phase2-baseline
	for mode in frozen finetune_partial; do for seed in 42 43 44; do \
	  $(PYTHON) second-installment/scripts/src/compare_paired.py --model-a second-installment/scripts/results/kermany/$$mode/seed$$seed/predictions_kermany_test.csv --metrics-a second-installment/scripts/results/kermany/$$mode/seed$$seed/metrics.json --model-b second-installment/scripts/results/kermany/baseline_random_forest/predictions_kermany_test.csv --metrics-b second-installment/scripts/results/kermany/baseline_random_forest/metrics.json --out second-installment/scripts/results/kermany/paired_$${mode}_seed$${seed}_vs_random_forest.json || exit 1; \
	done; done

# ---------------------------------------------------------------------------
# Fase 3: traslado a NIH ChestX-ray14 (adultos) y auditoría.
P3 = third-installment/scripts
P3_PARTITION = $(P3)/results/metadata/partition_nih.csv
P3_IMAGES = $(P3)/data/nih_images
P3_RESULTS = $(P3)/results/nih
KERMANY_RESULTS = second-installment/scripts/results/kermany
.PHONY: phase3-data phase3-run phase3-audit phase3-report phase3

phase3-data:
	bash $(P3)/download_nih.sh
	$(PYTHON) $(P3)/src/prepare_nih.py --metadata $(P3)/data/metadata/Data_Entry_2017_v2020.csv --bbox $(P3)/data/metadata/BBox_List_2017.csv --out $(P3)/results/metadata
	$(PYTHON) $(P3)/src/fetch_nih_images.py --partition $(P3_PARTITION) --out $(P3_IMAGES)

phase3-run:
	TORCH_HOME=/tmp/phase2-torch $(PYTHON) $(P3)/src/run_phase3.py --partition $(P3_PARTITION) --images-root $(P3_IMAGES) --config second-installment/scripts/configs/densenet121.json --kermany-results $(KERMANY_RESULTS) --results $(P3_RESULTS)

phase3-audit:
	$(PYTHON) $(P3)/src/run_audit.py --partition $(P3_PARTITION) --images-root $(P3_IMAGES) --bbox $(P3)/data/metadata/BBox_List_2017.csv --results $(P3_RESULTS) --kermany-results $(KERMANY_RESULTS) --kermany-partition second-installment/scripts/results/metadata/partition_kermany.csv --kermany-images first-installment/scripts/data/chest_xray --figures third-installment/report/figures

phase3-report:
	MPLCONFIGDIR=/tmp/matplotlib-phase3 $(PYTHON) $(P3)/src/summarize_phase3.py --results $(P3_RESULTS) --kermany-results $(KERMANY_RESULTS) --figures third-installment/report/figures --generated third-installment/report/generated

phase3: phase3-data phase3-run phase3-audit phase3-report

# Brecha de la réplica: punto de operación y partición por imagen (como en Rahman et al.).
.PHONY: phase3-replication
phase3-replication:
	$(PYTHON) $(P3)/src/replication_gap.py --make-image-split second-installment/scripts/data_manifest_kermany.csv --image-split-out $(P3)/results/metadata/partition_kermany_image_level.csv
	for seed in 42 43 44; do test -f $(P3)/results/kermany_image_split/finetune_partial/seed$$seed/metrics.json || \
	  TORCH_HOME=/tmp/phase2-torch $(PYTHON) second-installment/scripts/src/train.py --partition $(P3)/results/metadata/partition_kermany_image_level.csv --images-root first-installment/scripts/data/chest_xray --config second-installment/scripts/configs/densenet121.json --mode finetune_partial --seed $$seed --out $(P3)/results/kermany_image_split/finetune_partial/seed$$seed || exit 1; done
	$(PYTHON) $(P3)/src/replication_gap.py --kermany-results $(KERMANY_RESULTS) --image-split-results $(P3)/results/kermany_image_split --out third-installment/report/generated

# Todo el cómputo de la entrega final, en orden. Es reanudable: cada etapa omite
# las corridas completas, así que tras un corte basta con volver a ejecutarlo.
.PHONY: final
final:
	$(MAKE) phase2-train
	$(MAKE) phase2-compare-rf-finetune
	$(MAKE) phase2-figures
	$(MAKE) phase3-data
	$(MAKE) phase3-run
	$(MAKE) phase3-audit
	$(MAKE) phase3-report
	$(MAKE) phase3-replication

# Lo mismo, desatendido: corre en segundo plano bajo un supervisor que lo
# reinicia si se cae o se queda sin actividad. Tras apagar el equipo, basta con
# volver a ejecutar `make resume`; `make status` muestra en qué va.
.PHONY: resume status
resume:
	@mkdir -p logs
	@if [ -f logs/watchdog.pid ] && grep -qs supervise_final.sh /proc/$$(cat logs/watchdog.pid)/cmdline; then \
	  echo "Ya hay una supervisión activa (pid $$(cat logs/watchdog.pid))."; \
	else \
	  rm -f logs/ALERTA.txt; \
	  nohup setsid bash $(P3)/supervise_final.sh > /dev/null 2>&1 < /dev/null & \
	  echo "Supervisión iniciada. Progreso: make status"; \
	fi

status:
	@echo "Etapas terminadas:"; ls logs/stages 2>/dev/null | sed 's/\.done$$//; s/^/  /'
	@test ! -f logs/ALERTA.txt || { echo "*** ALERTA ***"; cat logs/ALERTA.txt; }
	@echo "Supervisor:"; tail -n 4 logs/watchdog.log 2>/dev/null | sed 's/^/  /'
	@echo "Últimas líneas:"; tail -n 4 logs/final.log 2>/dev/null | cut -c1-160 | sed 's/^/  /'
	@nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /' || true
