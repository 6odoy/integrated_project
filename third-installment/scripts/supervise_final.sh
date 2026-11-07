#!/usr/bin/env bash
# Supervisa run_final.sh. Lo arranca y lo reinicia si muere o si pasan STALL_MIN
# minutos sin que se escriba nada en el log ni en los resultados; las etapas son
# reanudables, así que un reinicio solo repite la corrida interrumpida. Tras
# MAX_RESTARTS reinicios seguidos sin que termine ninguna etapa, se detiene y deja
# logs/ALERTA.txt: un error que se repite no se arregla reintentando.
#
# Los procesos se identifican por archivo PID y /proc/<pid>/cmdline, no con
# `pgrep -f`: ese patrón también coincide con el shell que lanzó el script.
set -uo pipefail
cd "$(dirname "$0")/../.."
STALL_MIN=${STALL_MIN:-30}
CHECK_S=${CHECK_S:-120}
MAX_RESTARTS=${MAX_RESTARTS:-3}
RUNNER=third-installment/scripts/run_final.sh
WATCH=(logs/final.log logs/stages second-installment/scripts/results
       third-installment/scripts/results third-installment/report)
mkdir -p logs/stages

log() { echo "[$(date '+%F %T')] $*" >> logs/watchdog.log; }

runner_alive() {
  local pid
  pid=$(cat logs/final.pid 2>/dev/null) && [[ -n $pid ]] && grep -qs run_final.sh "/proc/$pid/cmdline"
}

start_runner() {
  # Sin control de trabajos el hijo no lidera su grupo, así que setsid no se
  # bifurca: $! es el PID del ejecutor y también el de su grupo de procesos.
  setsid bash "$RUNNER" >> logs/final.log 2>&1 < /dev/null &
  echo $! > logs/final.pid
  log "ejecutor iniciado (pid $!)"
}

stop_runner() {
  local pid
  pid=$(cat logs/final.pid)
  kill -TERM -- "-$pid" 2>/dev/null
  sleep 15
  kill -KILL -- "-$pid" 2>/dev/null
}

recent_activity() {
  local since
  since=$(date -d "$STALL_MIN minutes ago" '+%F %T')
  [[ -n $(find "${WATCH[@]}" -type f -newermt "$since" -print -quit 2>/dev/null) ]]
}

stages_done() { find logs/stages -name '*.done' | wc -l; }

echo $$ > logs/watchdog.pid
log "supervisión iniciada (sin actividad > ${STALL_MIN} min = atasco; revisión cada ${CHECK_S} s)"
restarts=0
done_before=$(stages_done)
runner_alive || start_runner

while sleep "$CHECK_S"; do
  if [[ -f logs/stages/all.done ]]; then
    log "todo completo; fin de la supervisión"
    break
  fi
  done_now=$(stages_done)
  if (( done_now > done_before )); then
    restarts=0
    done_before=$done_now
  fi
  if runner_alive; then
    recent_activity && continue
    log "SIN ACTIVIDAD durante ${STALL_MIN} min; diagnóstico:"
    { tail -n 15 logs/final.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
      ps -o pid,etime,cmd -g "$(cat logs/final.pid)"; } >> logs/watchdog.log 2>&1
    stop_runner
  else
    log "el ejecutor terminó sin completar; últimas líneas:"
    tail -n 15 logs/final.log >> logs/watchdog.log
  fi
  if (( restarts >= MAX_RESTARTS )); then
    log "ALERTA: ${MAX_RESTARTS} reinicios sin avanzar ninguna etapa; requiere intervención"
    { echo "El cómputo final se detuvo $(date '+%F %T')."; echo "Revisa logs/watchdog.log y logs/final.log."; } > logs/ALERTA.txt
    break
  fi
  restarts=$((restarts + 1))
  log "reinicio ${restarts}/${MAX_RESTARTS}"
  start_runner
done
rm -f logs/watchdog.pid
