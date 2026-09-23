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
from .model import (
    AnalysisParams,
    Fit,
    blur_map,
    brightness_matched_strength,
    fit_shadow,
    interpolate_aperture,
    residual,
    split_broad_component,
    strength_field,
)

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
    per_channel: bool = True  # intensidade medida separadamente em B, G e R
    broad_brightness_matched: bool = True  # manchas largas: só corrige com evidência de brilho parecido
    broad_touchup: bool = True  # ajuste final de cada mancha larga contra a vizinhança (sobra marrom ou branca)
    touchup_min: float = 0.02  # atenuação (log, desfocada) que define uma mancha larga para o ajuste final
    touchup_ring: int = 20  # vizinhança (px de análise) usada como referência no ajuste final
    touchup_max: float = 0.08  # teto do ajuste final, em log (≈ ±8%)


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
    if params.strength is None and params.broad_touchup and info is not None:
        out = _broad_touchup(out, dm, params, analysis, info.sigma)
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
        if params.per_channel:
            # A cor da sombra depende da luz da cena (a mancha marrom rouba mais azul sob céu azul do que
            # sob luz de lâmpada): mede a intensidade por canal e por região, puxada para o k de luminância.
            k_field = np.dstack([strength_field(res[..., c], a[..., c], k) for c in range(3)])
        else:
            k_field = strength_field(res, a, k)

    k3 = k_field if k_field.ndim == 3 else k_field[..., None]
    if params.strength is None and params.broad_brightness_matched:
        # Manchas largas (a marrom) também espalham luz: escurecem fundo claro e clareiam fundo escuro.
        # Os filamentos seguem o modelo acima; as manchas usam só evidência de brilho parecido.
        thin, broad = split_broad_component(a)
        small = cv2.resize(img_bgr, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
        log_lum = cv2.GaussianBlur(np.log(small.mean(axis=2) + 1.0), (0, 0), 3)
        k_broad = np.dstack([brightness_matched_strength(res[..., c], broad[..., c], log_lum) for c in range(3)])
        log_gain = k3 * thin + k_broad * broad
    else:
        log_gain = k3 * a
    log_gain = np.clip(log_gain, -params.max_gain, params.max_gain)
    h, w = img_bgr.shape[:2]
    gain = cv2.resize(np.exp(log_gain), (w, h), interpolation=cv2.INTER_CUBIC)
    out = np.clip(img_bgr.astype(np.float32) * gain + 0.5, 0, 255).astype(np.uint8)
    return out, CorrectionInfo(sigma, k, measured, fit)


def _broad_touchup(
    img_bgr: np.ndarray, dm: DefectMask, params: CorrectionParams, analysis: AnalysisParams, sigma: float
) -> np.ndarray:
    """Ajuste final de cada mancha larga, por canal, contra a vizinhança imediata.

    As passadas usam uma relação mancha/filamento vinda da calibração; sob outra luz a
    mancha marrom pode sobrar (marrom) ou passar do ponto (branca/azulada). Aqui, para
    cada mancha e canal, ajusta ``resíduo = offset + d·forma`` nos pixels lisos em volta
    e desfaz ``d`` com sinal livre, encolhido quando a medida é incerta.
    """
    _, broad_sharp = split_broad_component(dm.fungus_map)
    shape = blur_map(broad_sharp, sigma).mean(axis=2)
    n, labels = cv2.connectedComponents((shape > params.touchup_min).astype(np.uint8))
    if n <= 1:
        return img_bgr
    res = residual(img_bgr, analysis)
    ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * params.touchup_ring + 1,) * 2)
    log_gain = np.zeros(res.shape, np.float32)
    for i in range(1, n):
        comp = labels == i
        region = cv2.dilate(comp.astype(np.uint8), ring) > 0
        # Forma desta mancha só (suave, sem degrau), normalizada para 1 no pico.
        t = shape * cv2.GaussianBlur(comp.astype(np.float32), (0, 0), 2)
        t /= max(float(t.max()), 1e-6)
        for c in range(3):
            ok = region & ~np.isnan(res[..., c])
            core = ok & (t > 0.5)
            if core.sum() < 30 or (ok & (t < 0.05)).sum() < 60:
                continue  # sem área lisa na mancha ou em volta: não mexe
            x = t[ok]
            y = res[..., c][ok]
            xc, yc = x - x.mean(), y - y.mean()
            sxx = float(xc @ xc)
            d = float(xc @ yc) / sxx
            err = yc - d * xc
            se = float(np.sqrt((err @ err) / max(len(x) - 2, 1) / sxx))
            t2 = (d / se) ** 2 if se > 0 else 0.0
            d *= t2 / (t2 + 16.0)
            log_gain[..., c] -= np.clip(d, -params.touchup_max, params.touchup_max) * t
    if not log_gain.any():
        return img_bgr
    h, w = img_bgr.shape[:2]
    gain = cv2.resize(np.exp(log_gain), (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(img_bgr.astype(np.float32) * gain + 0.5, 0, 255).astype(np.uint8)
