# Segunda entrega — Fase 2

Pipeline reproducible de réplica reducida en Kermany y traslado con transferencia a BRAX. Se usa DenseNet121 en lugar de DenseNet201 porque el equipo disponible no tiene GPU; esa reducción está declarada en el informe.

```bash
make phase2-prepare
make phase2-train              # 2 configuraciones × 3 semillas; puede tardar varias horas en CPU
make phase2-figures
```

Cada corrida guarda historial, checkpoint, configuración, predicciones, métricas, IC bootstrap y umbral congelado en `scripts/results/kermany/<modo>/seed<semilla>/`. Las figuras y la tabla LaTeX se generan desde esos artefactos: nunca incluyen números inventados.

Los checkpoints no se versionan. Si existen métricas pero no `best_checkpoint.pt`, `make phase2-train` vuelve a ejecutar esa corrida para regenerarlo; es necesario antes de adaptar a BRAX.

La comparación contra la línea base se hace sobre **las mismas imágenes de prueba** y con bootstrap pareado por paciente; no se decide por el solapamiento de intervalos marginales. Para generarla (primero debe existir la corrida `finetune_partial/seed42`):

```bash
make phase2-baseline
make phase2-compare-rf-finetune
```

El primer comando guarda el Random Forest y sus predicciones en `scripts/results/kermany/baseline_random_forest/`. El segundo produce una comparación pareada por cada semilla; un IC de la diferencia A−B que excluya cero es la evidencia correspondiente. El umbral de cada modelo se fija solo con validación. Si su especificidad observada en test baja de 0,80, `sensitivity_at_specificity_minimum` se reporta como nula/no aplicable: no se debe presentar como sensibilidad que satisfaga la restricción.

Tras descargar BRAX legalmente, el protocolo se fija antes de entrenar: `prepare_brax.py` conserva solo PA/AP y etiquetas binarias inequívocas (`Pneumonia=1, No Finding=0` frente a `Pneumonia=0, No Finding=1`) y asigna pacientes, con semilla 42, a `adapt_train/adapt_val/external_test` en proporción 70/15/15. El último conjunto permanece aislado de la selección de época, umbral y modelo.

```bash
python second-installment/scripts/src/prepare_brax.py \
  --metadata <metadata_brax.csv> --out <brax_partition.csv>

# Ejecute ambos modos contra el mismo external_test; compare sus resultados.
python second-installment/scripts/src/adapt_brax.py \
  --manifest <brax_partition.csv> --images-root <imagenes_brax> \
  --source-run second-installment/scripts/results/kermany/finetune_partial/seed42 \
  --config second-installment/scripts/configs/densenet121.json \
  --mode frozen --out <resultado_brax_frozen>
python second-installment/scripts/src/adapt_brax.py \
  --manifest <brax_partition.csv> --images-root <imagenes_brax> \
  --source-run second-installment/scripts/results/kermany/finetune_partial/seed42 \
  --config second-installment/scripts/configs/densenet121.json \
  --mode finetune_partial --out <resultado_brax_parcial>
```

`prepare_brax.py` guarda el flujo de filtros, el resumen de particiones y el protocolo JSON. `adapt_brax.py` guarda historial, checkpoint, predicciones y bootstrap por paciente; nunca reentrena ni selecciona usando `external_test`.
