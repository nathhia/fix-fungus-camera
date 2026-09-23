"""Correção de uma imagem a partir da máscara estática.

Métodos:

* ``inpaint`` - ``cv2.inpaint`` em toda a máscara (fungo + pixels quentes).
  Bom para defeitos opacos e pequenos. Em filamentos largos sobre textura
  (folhagem, rostos), inventa conteúdo e borra.
* ``flatfield`` - o fungo é uma sombra *semitransparente*: a luz ainda passa,
  só atenuada. Dividir pela atenuação medida (ganho = ``exp(A)``) recupera
  a textura original sob o filamento, em vez de inventá-la.
* ``hybrid`` (padrão) - *flat-field* no fungo + ``cv2.inpaint`` nos pixels
  quentes e nos núcleos quase opacos do fungo, onde o ganho amplificaria
  ruído demais.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .mask import DefectMask, MaskParams, fungus_residual

log = logging.getLogger(__name__)


class Method(str, Enum):
    INPAINT = "inpaint"
    FLATFIELD = "flatfield"
    HYBRID = "hybrid"


class InpaintAlgorithm(str, Enum):
    TELEA = "telea"
    NS = "ns"

    @property
    def cv_flag(self) -> int:
        return cv2.INPAINT_TELEA if self is InpaintAlgorithm.TELEA else cv2.INPAINT_NS


@dataclass(frozen=True)
class CorrectionParams:
    method: Method = Method.HYBRID
    algorithm: InpaintAlgorithm = InpaintAlgorithm.TELEA
    inpaint_radius: int = 5
    core_threshold: float = 0.25  # hybrid: atenuação (log) acima da qual o fungo é tratado como opaco
    auto_strength: bool = True  # ajusta a intensidade do flat-field por foto (abertura/zoom mudam a sombra)
    strength: float = 1.0  # multiplicador manual da atenuação (usado se auto_strength=False)


def estimate_strength(img_bgr: np.ndarray, fungus_map: np.ndarray, analysis: MaskParams) -> float | None:
    """Quanto o fungo escurece *esta* foto em relação ao mapa médio.

    Mínimos quadrados ``residual ≈ -k·A`` nas regiões lisas da própria foto.
    Retorna ``None`` se a foto não tiver área lisa suficiente sobre o fungo.
    """
    r = fungus_residual(img_bgr, analysis)
    a = cv2.resize(fungus_map, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_AREA)
    valid = ~np.isnan(r) & (a > analysis.fungus_threshold)
    if valid.sum() < 500:
        return None
    rv, av = r[valid], a[valid]
    return float(np.clip(-(rv * av).sum() / (av * av).sum(), 0.3, 2.0))


def flatfield(img_bgr: np.ndarray, fungus_map: np.ndarray, strength: float) -> np.ndarray:
    gain = np.exp(strength * fungus_map)[..., None]
    return np.clip(img_bgr.astype(np.float32) * gain + 0.5, 0, 255).astype(np.uint8)


def correct_image(
    img_bgr: np.ndarray,
    dm: DefectMask,
    params: CorrectionParams,
    analysis: MaskParams | None = None,
) -> tuple[np.ndarray, float | None]:
    """Aplica a correção. Retorna ``(imagem, intensidade do flat-field usada ou None)``."""
    if img_bgr.shape[:2] != dm.shape:
        raise ValueError(f"resolução {img_bgr.shape[:2]} difere da máscara {dm.shape}")

    if params.method is Method.INPAINT:
        return cv2.inpaint(img_bgr, dm.mask, params.inpaint_radius, params.algorithm.cv_flag), None

    k = params.strength
    if params.auto_strength:
        est = estimate_strength(img_bgr, dm.fungus_map, analysis or MaskParams())
        k = est if est is not None else params.strength
    out = flatfield(img_bgr, dm.fungus_map, k)

    if params.method is Method.FLATFIELD:
        return out, k

    core = (dm.fungus_map * k > params.core_threshold).astype(np.uint8) * 255
    core = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    to_inpaint = cv2.bitwise_or(core, dm.hotpixel_mask)
    if to_inpaint.any():
        out = cv2.inpaint(out, to_inpaint, params.inpaint_radius, params.algorithm.cv_flag)
    return out, k
