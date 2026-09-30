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
    edge_bin_width: float = 0.3  # fundo local respeita bordas: faixas de brilho em log (0 = desliga)


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
    guide = cv2.GaussianBlur(lum, (0, 0), 4)  # brilho da região, sem os detalhes finos (nem a sombra)
    bg_lum = _edge_aware_blur(lum[..., None], flat, guide, sigma, p.edge_bin_width)[..., 0]
    # 2ª passada sem a própria sombra, para ela não puxar o fundo para baixo.
    w = flat * ((lum - bg_lum) > -dark_threshold)
    out = logi - _edge_aware_blur(logi, w, guide, sigma, p.edge_bin_width)
    out[flat == 0] = np.nan
    return out, texture


def _edge_aware_blur(
    values: np.ndarray, weight: np.ndarray, guide: np.ndarray, sigma: float, bin_width: float
) -> np.ndarray:
    """Fundo local que não atravessa bordas fortes (horizonte, parede x móvel).

    Convolução normalizada separada por faixas de brilho do ``guide`` (log):
    cada pixel é comparado só com vizinhos de brilho parecido, então o mar logo
    abaixo do céu não parece "sombreado" por causa do céu claro ao lado.
    ``bin_width`` <= 0 desliga (blur comum).
    """
    if bin_width <= 0:
        return np.dstack([_normalized_blur(values[..., c], weight, sigma) for c in range(values.shape[2])])
    lo, hi = np.percentile(guide, 0.5), np.percentile(guide, 99.5)
    num = np.zeros(values.shape, np.float32)
    den = np.zeros(guide.shape, np.float32)
    for center in np.arange(lo, hi + bin_width, bin_width):
        member = np.clip(1.0 - np.abs(guide - center) / bin_width, 0.0, 1.0).astype(np.float32)
        if not member.any():
            continue
        wb = weight * member
        wsum = cv2.GaussianBlur(wb, (0, 0), sigma)
        for c in range(values.shape[2]):
            blurred = cv2.GaussianBlur(values[..., c] * wb, (0, 0), sigma) / np.maximum(wsum, 1e-6)
            num[..., c] += member * blurred * (wsum > 1e-3)
        den += member * (wsum > 1e-3)
    fallback = np.dstack([_normalized_blur(values[..., c], weight, sigma) for c in range(values.shape[2])])
    has = den > 1e-6
    out = fallback.copy()
    out[has] = num[has] / den[has][:, None]
    return out


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


def split_broad_component(a: np.ndarray, radius: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Separa manchas largas (ex.: a mancha marrom) dos filamentos finos por abertura morfológica.

    Retorna ``(filamentos, manchas)`` com ``filamentos + manchas == a``.
    """
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    broad = cv2.morphologyEx(a, cv2.MORPH_OPEN, disk)
    broad = cv2.GaussianBlur(broad, (0, 0), radius / 3)  # sem degrau na borda da mancha
    broad = np.minimum(broad, a)
    return a - broad, broad


def brightness_matched_strength(
    res: np.ndarray,
    x: np.ndarray,
    log_lum: np.ndarray,
    tile: int = 40,
    bin_width: float = 0.25,
    prior_weight: float = 0.3,
) -> np.ndarray:
    """Intensidade local usando só evidência de pixels com brilho parecido; sem evidência, 0.

    Para defeitos cujo efeito depende do que está atrás (a mancha marrom escurece
    o céu claro mas clareia o mar escuro, por espalhar luz), a medida feita no céu
    não vale para o mar. Cada faixa de brilho (``bin_width`` em log) tem seu próprio
    ajuste; a saída mistura as faixas vizinhas suavemente.
    """
    r = res
    w_flat = (~np.isnan(r)).astype(np.float32)
    y = np.nan_to_num(-r).astype(np.float32)
    box = (tile, tile)
    lo, hi = np.percentile(log_lum, 1), np.percentile(log_lum, 99)
    centers = np.arange(lo, hi + bin_width, bin_width)
    num = np.zeros_like(x, dtype=np.float32)
    den = np.zeros_like(x, dtype=np.float32)
    sxx_all = cv2.boxFilter(w_flat * x * x, -1, box, normalize=False)
    informative = sxx_all[sxx_all > 0]
    lam = prior_weight * float(np.percentile(informative, 90)) if informative.size else 1.0
    for c in centers:
        member = np.clip(1.0 - np.abs(log_lum - c) / bin_width, 0.0, 1.0).astype(np.float32)  # triangular
        w = w_flat * member
        sxy = cv2.boxFilter(w * x * y, -1, box, normalize=False)
        sxx = cv2.boxFilter(w * x * x, -1, box, normalize=False)
        k_c = sxy / (sxx + lam)  # prior 0: sem evidência, não corrige
        num += member * k_c
        den += member
    k = num / np.maximum(den, 1e-6)
    return cv2.GaussianBlur(np.clip(k, -2.0, 4.0).astype(np.float32), (0, 0), tile / 4)
