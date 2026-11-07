"""Descarga solo las imágenes NIH de la partición, sin bajar los ~45 GB.

Las imágenes originales de NIH (nihcc.app.box.com/v/ChestXray-NIHCC) se leen
del espejo público alkzar90/NIH-Chest-X-ray-dataset en Hugging Face, que las
reempaqueta en 12 ZIP. Un ZIP permite acceso aleatorio: se lee su directorio
central con peticiones HTTP por rango y se extrae cada PNG por separado.
``zipfile`` verifica el CRC32 de cada miembro, así que un archivo corrupto
falla en lugar de quedar en disco. Es reanudable: omite los PNG ya presentes.
"""
from __future__ import annotations

import argparse
import io
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

MIRROR = "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/images/images_{:03d}.zip"


class HttpRangeFile(io.RawIOBase):
    """Archivo de solo lectura sobre HTTP con caché por bloques, suficiente para ``zipfile``."""

    def __init__(self, url: str, block: int = 1 << 20):
        self.source, self.block, self.pos, self.cache = url, block, 0, {}
        self._resolve()

    def _resolve(self):
        """Sigue la redirección al CDN una vez; la URL firmada se renueva si expira."""
        request = urllib.request.Request(self.source, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            self.url, self.size = response.geturl(), int(response.headers["Content-Length"])

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def _range(self, start: int, end: int) -> bytes:
        for attempt in range(5):
            try:
                request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = response.read()
                if len(data) == end - start:
                    return data
            except OSError:
                pass
            time.sleep(2 ** attempt)
            self._resolve()
        raise IOError(f"No se pudo leer {self.url} [{start}, {end})")

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        # Lecturas grandes (un PNG completo) van directas; las pequeñas usan bloques en caché.
        if n > self.block:
            data = self._range(self.pos, self.pos + n)
        else:
            chunks, start = [], self.pos
            while start < self.pos + n:
                index = start // self.block
                if index not in self.cache:
                    if len(self.cache) > 64:
                        self.cache.clear()
                    self.cache[index] = self._range(index * self.block, min((index + 1) * self.block, self.size))
                offset = start - index * self.block
                piece = self.cache[index][offset:offset + self.pos + n - start]
                chunks.append(piece)
                start += len(piece)
            data = b"".join(chunks)
        self.pos += len(data)
        return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partition", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="Directorio donde quedan los PNG")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    wanted = set(pd.read_csv(args.partition).image_path)
    args.out.mkdir(parents=True, exist_ok=True)
    pending = {name for name in wanted if not (args.out / name).exists()}
    print(f"{len(wanted)} imágenes en la partición; {len(pending)} por descargar", flush=True)
    for archive in range(1, 13):
        if not pending:
            break
        url = MIRROR.format(archive)
        # El espejo incluye metadatos de macOS (__MACOSX/._*) que no son imágenes.
        members = {Path(info.filename).name: info for info in zipfile.ZipFile(HttpRangeFile(url)).infolist()
                   if not info.is_dir() and not info.filename.startswith("__MACOSX")}
        todo = sorted(pending & members.keys())
        print(f"images_{archive:03d}.zip: {len(members)} miembros, {len(todo)} seleccionados", flush=True)

        local = threading.local()

        def fetch(name: str) -> str:
            # Un lector por hilo: zipfile no es seguro con un mismo archivo compartido.
            if not hasattr(local, "handle"):
                local.handle = zipfile.ZipFile(HttpRangeFile(url))
            data = local.handle.read(members[name])  # lanza BadZipFile si el CRC32 no coincide
            temporary = args.out / (name + ".part")
            temporary.write_bytes(data)
            temporary.rename(args.out / name)
            return name

        with ThreadPoolExecutor(args.workers) as pool:
            for done, future in enumerate(as_completed(pool.submit(fetch, n) for n in todo), 1):
                pending.discard(future.result())
                if done % 200 == 0:
                    print(f"  {done}/{len(todo)}", flush=True)
    if pending:
        raise SystemExit(f"Faltan {len(pending)} imágenes que no aparecen en el espejo: {sorted(pending)[:5]}")
    print(f"Listo: {len(wanted)} imágenes verificadas por CRC32 en {args.out}")


if __name__ == "__main__":
    main()
