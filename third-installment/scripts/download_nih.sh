#!/usr/bin/env bash
# Descarga y verifica los metadatos NIH ChestX-ray14; las imágenes se bajan
# después, solo las de la partición, con src/fetch_nih_images.py.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data/metadata"
# Etiquetas: NIH Clinical Center (nihcc.app.box.com/v/ChestXray-NIHCC), versión v2020.
ENTRY_URL="https://nihcc.app.box.com/index.php?rm=box_download_shared_file&vanity_name=ChestXray-NIHCC&file_id=f_219760887468"
ENTRY_SHA256="c69a6daca3549af707ca9cacbdf9f9a7b6a9188e8c61157df653520bb72d8eb1"
# Cajas del radiólogo: copia idéntica del archivo NIH en el espejo de Hugging Face.
BBOX_URL="https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/BBox_List_2017.csv"
BBOX_SHA256="0bbfea9d4c4e9771481b3023b1bc9f0df9dea924453b12986beb29b0c4d0c95b"

command -v curl >/dev/null || { echo "Se requiere curl." >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "Se requiere sha256sum." >&2; exit 1; }
mkdir -p "$DATA_DIR"

fetch() {
    local url="$1" sha="$2" target="$3"
    if [[ -f "$target" ]] && echo "$sha  $target" | sha256sum --check --status; then
        echo "$(basename "$target") ya verificado."
        return
    fi
    echo "Descargando $(basename "$target")..."
    curl --fail --location --retry 3 --silent --show-error --output "$target" "$url"
    echo "$sha  $target" | sha256sum --check --status || { echo "SHA-256 inválido: $target" >&2; rm -f "$target"; exit 1; }
}

fetch "$ENTRY_URL" "$ENTRY_SHA256" "$DATA_DIR/Data_Entry_2017_v2020.csv"
fetch "$BBOX_URL" "$BBOX_SHA256" "$DATA_DIR/BBox_List_2017.csv"
echo "Metadatos NIH disponibles en $DATA_DIR"
