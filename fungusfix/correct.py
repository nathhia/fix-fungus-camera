"""Correção de uma imagem a partir da máscara estática.

Métodos:

* ``inpaint`` - ``cv2.inpaint`` em toda a máscara binária. Bom para
  defeitos opacos e pequenos; em filamentos largos sobre textura
  (folhagem, rostos, ondas) inventa conteúdo e borra.
* ``flatfield`` - a sombra é *semitransparente*: a luz passa, só atenuada.
  Dividir pela atenuação recupera a textura original em vez de inventá-la.
  O desfoque σ e a intensidade k da sombra são ajustados **por foto**, e k
  varia no espaço: onde a foto não mostra a sombra, a correção recua.
* ``hybrid`` (padrão) - ``flatfield`` + ``cv2.inpaint`` só nos pixels quentes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .mask import DefectMask
from .model import AnalysisParams, Fit, blur_map, fit_shadow, interpolate_aperture, residual, strength_field

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
    strength: float | None = None  # None = automático por foto; número = intensidade fixa
    blur: float | None = None  # None = automático; número = desfoque fixo (px de análise)
    max_gain: float = 0.4  # teto da correção em log (e^0.4 ≈ +49%), evita "estourar" um ponto
    unreliable_fraction: float = 0.5  # sem área lisa para medir: usa esta fração da intensidade típica
    sigma_window: float = 3.0  # com calibração: σ buscado só até ±isto em volta do previsto pela abertura
    max_k_factor: float = 4.0  # k medido limitado a este múltiplo do esperado (acima disso é cena, não sombra)
    passes: int = 2  # repete medir+corrigir: a 2ª passada pega a sombra que a 1ª subestimou


@dataclass(frozen=True)
class CorrectionInfo:
    sigma: float
    k: float
    measured: bool  # True = σ e k medidos nesta foto; False = valores típicos da máscara
    fit: Fit | None


def _shrink(fit: Fit) -> float:
    """Encolhe k em direção a 0 quando a medida é incerta (t baixo)."""
    t2 = fit.t * fit.t
    return max(0.0, fit.k) * t2 / (t2 + 16.0)


def _expected(dm: DefectMask, fnumber: float | None) -> tuple[float, float]:
    """(σ, k) esperados para esta foto: pela abertura, se a máscara foi calibrada; senão os típicos."""
    if dm.aperture_table and fnumber:
        return interpolate_aperture(dm.aperture_table, fnumber)
    return dm.prior_sigma, dm.prior_k


def correct_image(
    img_bgr: np.ndarray,
    dm: DefectMask,
    params: CorrectionParams,
    analysis: AnalysisParams,
    fnumber: float | None = None,
) -> tuple[np.ndarray, CorrectionInfo | None]:
    """Aplica a correção. Retorna ``(imagem, parâmetros da sombra usados ou None para inpaint)``.

    ``fnumber`` (abertura do EXIF) restringe o desfoque procurado ao previsto
    pela calibração, o que evita confundir variações lentas da cena com sombra.
    """
    if img_bgr.shape[:2] != dm.shape:
        raise ValueError(f"resolução {img_bgr.shape[:2]} difere da máscara {dm.shape}")

    if params.method is Method.INPAINT:
        return cv2.inpaint(img_bgr, dm.mask, params.inpaint_radius, params.algorithm.cv_flag), None

    out = img_bgr
    info: CorrectionInfo | None = None
    # Com intensidade fixa não há o que remedir: repetir só aplicaria a mesma correção duas vezes.
    passes = 1 if params.strength is not None else max(1, params.passes)
    for _ in range(passes):
        out, info = _flatfield_pass(out, dm, params, analysis, fnumber)
    if params.method is Method.HYBRID and dm.hotpixel_mask.any():
        out = cv2.inpaint(out, dm.hotpixel_mask, params.inpaint_radius, params.algorithm.cv_flag)
    return out, info


def _flatfield_pass(
    img_bgr: np.ndarray, dm: DefectMask, params: CorrectionParams, analysis: AnalysisParams, fnumber: float | None
) -> tuple[np.ndarray, CorrectionInfo]:
    res = residual(img_bgr, analysis)
    sigma_exp, k_exp = _expected(dm, fnumber)
    candidates = None
    if dm.aperture_table and fnumber:
        candidates = tuple(s for s in analysis.blur_sigmas if abs(s - sigma_exp) <= params.sigma_window)
    fit = fit_shadow(res, dm.fungus_map, analysis, sigmas=candidates or None)
    measured = fit.reliable
    sigma = params.blur if params.blur is not None else (fit.sigma if measured else sigma_exp)
    a = blur_map(dm.fungus_map, sigma)
    if params.strength is not None:
        k_field = np.full(a.shape[:2], params.strength, np.float32)
        k = params.strength
    else:
        # Medida confiável: usa, mas limitada a um múltiplo do esperado (k alto demais = cena, não sombra).
        k = min(_shrink(fit), params.max_k_factor * k_exp + 0.25) if measured else params.unreliable_fraction * k_exp
        k_field = strength_field(res, a, k)

    log_gain = np.minimum(k_field[..., None] * a, params.max_gain)
    h, w = img_bgr.shape[:2]
    gain = cv2.resize(np.exp(log_gain), (w, h), interpolation=cv2.INTER_CUBIC)
    out = np.clip(img_bgr.astype(np.float32) * gain + 0.5, 0, 255).astype(np.uint8)
    return out, CorrectionInfo(sigma, k, measured, fit)
