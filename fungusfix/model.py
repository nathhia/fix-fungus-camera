"""Modelo físico da sombra do fungo e seu ajuste por foto.

A sombra é modelada como ``foto = cena · exp(-k · G_σ ∗ A)``, com:

* ``A`` - atenuação fixa (log, por canal BGR), medida nas referências.
  Posição fixa para um dado zoom.
* ``σ`` - desfoque extra da sombra *nesta* foto. Muda com abertura e luz:
  a mesma câmera, no mesmo zoom, produz a sombra nítida numa foto e
  espalhada na seguinte.
* ``k`` - intensidade *nesta* foto (0 = sem sombra).

Tudo aqui trabalha na escala de análise (1/4 da resolução por padrão),
suficiente para uma sombra desfocada e muito mais rápida.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AnalysisParams:
    scale: float = 0.25
    background_sigma: float = 250.0  # px na resolução original; > maior defeito (mancha de ~280 px)
    flat_texture_max: float = 0.022  # textura fina máxima (log) para considerar a região lisa
    blur_sigmas: tuple[float, ...] = (0, 1, 2, 3, 4, 6, 8, 11, 14)  # desfoques testados (px de análise)


def _normalized_blur(values: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Blur gaussiano que ignora pixels com peso 0 (convolução normalizada)."""
    num = cv2.GaussianBlur(values * weight, (0, 0), sigma)
    den = cv2.GaussianBlur(weight, (0, 0), sigma)
    return num / np.maximum(den, 1e-6)


def analysis_shape(full_shape: tuple[int, int], p: AnalysisParams) -> tuple[int, int]:
    return round(full_shape[0] * p.scale), round(full_shape[1] * p.scale)


def date_stamp_mask(small_bgr: np.ndarray) -> np.ndarray:
    """Pixels do carimbo de data da câmera (texto laranja saturado), com margem."""
    hsv = cv2.cvtColor(small_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
    orange = (hsv[..., 0] >= 5) & (hsv[..., 0] <= 25) & (hsv[..., 1] > 150) & (hsv[..., 2] > 120)
    return cv2.dilate(orange.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0


def residual(
    img_bgr: np.ndarray, p: AnalysisParams, full_shape: tuple[int, int] | None = None, dark_threshold: float = 0.015
) -> np.ndarray:
    """``log(I) - log(fundo local)`` por canal, na escala de análise; NaN onde a foto não é lisa.

    ``full_shape`` é a resolução do sensor; uma foto menor (ex.: 640x480) é
    redimensionada para a mesma grade de análise. Retorna ``(h, w, 3)`` float32.
    Negativo = mais escuro que o entorno.
    """
    out, _ = residual_and_texture(img_bgr, p, full_shape, dark_threshold)
    return out


def residual_and_texture(
    img_bgr: np.ndarray, p: AnalysisParams, full_shape: tuple[int, int] | None = None, dark_threshold: float = 0.015
) -> tuple[np.ndarray, np.ndarray]:
    """Como :func:`residual`, e também o mapa de textura fina (para pesar pixels no ajuste)."""
    h, w = analysis_shape(full_shape or img_bgr.shape[:2], p)
    interp = cv2.INTER_AREA if img_bgr.shape[1] >= w else cv2.INTER_CUBIC
    small = cv2.resize(img_bgr, (w, h), interpolation=interp).astype(np.float32)
    logi = np.log(small + 1.0)
    lum = logi.mean(axis=2)
    gray = small.mean(axis=2)

    # Textura numa escala menor que a sombra: céu/parede ~ 0, folhagem >> 0.
    fine = lum - cv2.GaussianBlur(lum, (0, 0), 1.5)
    texture = np.sqrt(cv2.GaussianBlur(fine * fine, (0, 0), 6))
    flat = (texture < p.flat_texture_max) & (gray > 30) & (small.max(axis=2) < 250) & ~date_stamp_mask(small)
    flat = cv2.erode(flat.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)

    sigma = p.background_sigma * p.scale
    bg_lum = _normalized_blur(lum, flat, sigma)
    # 2ª passada sem a própria sombra, para ela não puxar o fundo para baixo.
    w = flat * ((lum - bg_lum) > -dark_threshold)
    out = np.empty_like(logi)
    for c in range(3):
        out[..., c] = logi[..., c] - _normalized_blur(logi[..., c], w, sigma)
    out[flat == 0] = np.nan
    return out, texture


def blur_map(a: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(a, (0, 0), sigma) if sigma > 0 else a


@dataclass(frozen=True)
class Fit:
    sigma: float
    k: float
    t: float  # "estatística t" de k: confiança de que a sombra foi medida (≈ sinal/ruído)
    n: int  # pixels lisos usados

    @property
    def reliable(self) -> bool:
        return self.t >= 4.0 and self.n >= 2000


def fit_shadow(
    res: np.ndarray,
    a: np.ndarray,
    p: AnalysisParams,
    support: float = 0.003,
    sigmas: tuple[float, ...] | None = None,
) -> Fit:
    """Ajusta (σ, k) para que ``-k · G_σ ∗ A`` explique o resíduo da foto.

    Mínimos quadrados em luminância, só nos pixels lisos próximos ao defeito.
    A incerteza de k considera que pixels vizinhos são correlacionados
    (cada "amostra independente" ~ 5x5 px de análise).
    """
    r = np.nanmean(res, axis=2) if res.ndim == 3 else res
    a_lum = a.mean(axis=2) if a.ndim == 3 else a
    # Mesmo conjunto de pixels para todos os σ (o suporte do mapa mais espalhado), senão os erros não se comparam.
    v = ~np.isnan(r) & (blur_map(a_lum, max(p.blur_sigmas)) > support)
    n = int(v.sum())
    best = Fit(0.0, 0.0, 0.0, n)
    if n < 300:
        return best
    y = -r[v]
    best_err = np.inf
    for s in sigmas or p.blur_sigmas:
        x = blur_map(a_lum, s)[v]
        sxx = float((x * x).sum())
        if sxx <= 0:
            continue
        k = float((x * y).sum()) / sxx
        err = float(((y - k * x) ** 2).mean())
        if err < best_err:
            se = np.sqrt(err / sxx * 25.0)
            best_err, best = err, Fit(float(s), k, k / se if se > 0 else 0.0, n)
    return best


def strength_field(
    res: np.ndarray,
    a_blurred: np.ndarray,
    k_global: float,
    tile: int = 40,
    prior_fraction: float = 0.5,
    prior_weight: float = 0.3,
) -> np.ndarray:
    """Intensidade k variando no espaço, estimada em janelas de ``tile`` px (escala de análise).

    Cada janela faz seu próprio mínimos quadrados nos pixels lisos, puxado
    para ``prior_fraction · k_global`` onde há pouca evidência. Onde a foto
    não mostra a sombra (textura, objeto na frente), a correção fica
    conservadora em vez de clarear um ponto que não estava escuro.
    """
    r = np.nanmean(res, axis=2) if res.ndim == 3 else res
    x = a_blurred.mean(axis=2) if a_blurred.ndim == 3 else a_blurred
    w = (~np.isnan(r)).astype(np.float32)
    y = np.nan_to_num(-r).astype(np.float32)
    box = (tile, tile)
    sxy = cv2.boxFilter(w * x * y, -1, box, normalize=False)
    sxx = cv2.boxFilter(w * x * x, -1, box, normalize=False)
    informative = sxx[sxx > 0]
    lam = prior_weight * float(np.percentile(informative, 90)) if informative.size else 1.0
    k0 = prior_fraction * k_global
    k = (sxy + lam * k0) / (sxx + lam)
    k = np.clip(k, 0.0, max(k_global, k0) * 1.5)
    return cv2.GaussianBlur(k.astype(np.float32), (0, 0), tile / 2)


@dataclass(frozen=True)
class ApertureEntry:
    fnumber: float
    sigma: float
    k: float


def interpolate_aperture(table: list[ApertureEntry], fnumber: float) -> tuple[float, float]:
    """(σ, k) previstos para uma abertura, interpolando em 1/F (proporcional ao cone de luz)."""
    pts = sorted(table, key=lambda e: 1.0 / e.fnumber)
    x = [1.0 / e.fnumber for e in pts]
    return (
        float(np.interp(1.0 / fnumber, x, [e.sigma for e in pts])),
        float(np.interp(1.0 / fnumber, x, [e.k for e in pts])),
    )
