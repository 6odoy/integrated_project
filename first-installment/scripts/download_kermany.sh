#!/usr/bin/env bash
# Descarga y verifica Chest X-Ray Pneumonia (Kermany) para ejecutar la línea base.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
ARCHIVE="$DATA_DIR/ChestXRay2017.zip"
URL="https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/f12eaf6d-6023-432f-acc9-80c9d7393433/file_downloaded"
SHA256="13efc055629733dbab07877f8b3c9f81097840dbcdaa326a8542322c2281ce36"

if [[ -d "$DATA_DIR/chest_xray" ]]; then
    echo "El dataset ya existe en $DATA_DIR/chest_xray; no se descarga de nuevo."
    exit 0
fi

command -v curl >/dev/null || { echo "Se requiere curl." >&2; exit 1; }
command -v unzip >/dev/null || { echo "Se requiere unzip." >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "Se requiere sha256sum." >&2; exit 1; }

mkdir -p "$DATA_DIR"
if [[ -f "$ARCHIVE" ]] && echo "$SHA256  $ARCHIVE" | sha256sum --check --status; then
    echo "Archivo descargado y verificado previamente."
else
    # Este servidor anuncia rangos, pero no siempre los respeta tras la redirección.
    # Un archivo parcial se descarta para impedir concatenar copias corruptas.
    if [[ -f "$ARCHIVE" ]]; then
        echo "Archivo parcial o inválido detectado; se descarga de nuevo."
        rm -f "$ARCHIVE"
    fi
    echo "Descargando ChestXRay2017.zip..."
    curl --fail --location --retry 3 --silent --show-error \
        --output "$ARCHIVE" "$URL"
    echo "$SHA256  $ARCHIVE" | sha256sum --check --status
fi
unzip -q "$ARCHIVE" -d "$DATA_DIR"
echo "Dataset disponible en $DATA_DIR/chest_xray"
