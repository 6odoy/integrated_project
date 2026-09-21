# Segunda entrega — Fase 2

Pipeline reproducible de transferencia para Kermany y evaluación externa congelada en BRAX. Se usa DenseNet121 en lugar de DenseNet201 porque el equipo disponible no tiene GPU; esa reducción está declarada en el informe.

```bash
make phase2-prepare
make phase2-train              # 2 configuraciones × 3 semillas; puede tardar varias horas en CPU
make phase2-figures
```

Cada corrida guarda historial, checkpoint, configuración, predicciones, métricas, IC bootstrap y umbral congelado en `scripts/results/kermany/<modo>/seed<semilla>/`. Las figuras y la tabla LaTeX se generan desde esos artefactos: nunca incluyen números inventados.

BRAX queda fuera de entrenamiento y selección. Tras descargarlo legalmente, normalice el CSV con `prepare_brax.py` y ejecute `evaluate_brax.py` contra una corrida finalizada. Los nombres de columnas se parametrizan para corresponder al CSV oficial.
