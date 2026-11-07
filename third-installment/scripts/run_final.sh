#!/usr/bin/env bash
# Ejecuta las etapas de `make final` en orden y marca cada una en logs/stages al
# terminar, para que un reinicio continúe desde la primera etapa sin marca. Para
# forzar que una etapa se repita, borra su marca: rm logs/stages/<etapa>.done
set -euo pipefail
cd "$(dirname "$0")/../.."
# Sin búfer: el supervisor mide la actividad por lo que se escribe en el log.
export PYTHONUNBUFFERED=1
STAGES=(phase2-train phase2-compare-rf-finetune phase2-figures phase3-data
        phase3-run phase3-audit phase3-report phase3-replication)
mkdir -p logs/stages
for stage in "${STAGES[@]}"; do
  if [[ -f logs/stages/$stage.done ]]; then
    echo "[$(date '+%F %T')] [omitida] $stage"
    continue
  fi
  echo "[$(date '+%F %T')] [inicia] $stage"
  make "$stage"
  touch "logs/stages/$stage.done"
  echo "[$(date '+%F %T')] [termina] $stage"
done
touch logs/stages/all.done
echo "[$(date '+%F %T')] TODO COMPLETO"
