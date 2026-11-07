"""Piezas comunes de la auditoría: modelo, imágenes en caché, puntajes y Grad-CAM."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nih_common import XrayDataset, transforms_for  # noqa: F401  (también fija sys.path de Fase 2)
from train import MEAN, STD, build_model, get_device

_MEAN = torch.tensor(MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(STD).view(1, 3, 1, 1)


def load_model(run_dir: Path, device):
    config = json.loads((run_dir / "run_config.json").read_text())
    model = build_model(config["mode"], pretrained=False)
    model.load_state_dict(torch.load(run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=True))
    threshold = json.loads((run_dir / "metrics.json").read_text())["threshold"]
    return model.to(device).eval(), threshold


def cached_images(frame, images_root, size=224) -> np.ndarray:
    """uint8 (N, size, size, 3) con el mismo Resize que el entrenamiento."""
    return XrayDataset(frame, images_root, transforms_for(False), cache_resize=size).cache


def to_tensor(images_uint8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(images_uint8)).permute(0, 3, 1, 2).float() / 255
    return (x - _MEAN) / _STD


@torch.inference_mode()
def scores(model, images_uint8: np.ndarray, device, batch_size=32) -> np.ndarray:
    out = []
    for start in range(0, len(images_uint8), batch_size):
        x = to_tensor(images_uint8[start:start + batch_size]).to(device)
        out.append(torch.sigmoid(model(x)).flatten().cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def gradcam(model, images_uint8: np.ndarray, device, batch_size=16) -> np.ndarray:
    """Grad-CAM sobre la salida de ``features`` (tras norm5 y ReLU), reescalado a la entrada.

    Se recompone el forward de DenseNet sin la ReLU in-place de torchvision para
    que el gradiente respecto a las activaciones sea el de la capa que se pondera.
    """
    maps = []
    size = images_uint8.shape[1]
    for start in range(0, len(images_uint8), batch_size):
        x = to_tensor(images_uint8[start:start + batch_size]).to(device)
        with torch.enable_grad():
            activations = F.relu(model.features(x))
            activations.retain_grad()
            logits = model.classifier(torch.flatten(F.adaptive_avg_pool2d(activations, 1), 1)).flatten()
            logits.sum().backward()
        weights = activations.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)
        maps.append(cam[:, 0].detach().cpu().numpy())
        model.zero_grad(set_to_none=True)
    return np.concatenate(maps)


def normalized(cam: np.ndarray) -> np.ndarray:
    """Mapa como distribución de masa (suma 1); un mapa nulo queda uniforme."""
    total = cam.sum()
    return cam / total if total > 0 else np.full_like(cam, 1 / cam.size)


__all__ = ["load_model", "cached_images", "scores", "gradcam", "normalized", "get_device"]
