"""Fotos que não são o sensor inteiro em resolução total: zoom digital, 16:9/3:2/1:1 e tamanhos menores.

Toda foto desta câmera é um retângulo centrado no sensor, reamostrado para o tamanho do arquivo:

* resolução menor (M, S, 640x480): o sensor inteiro, reduzido;
* 16:9, 3:2, 1:1: um recorte centrado que ocupa a largura (ou a altura) inteira;
* zoom digital: um recorte centrado ``fator`` vezes menor, ampliado.

A correção é feita na escala do sensor, com o mapa recortado no mesmo retângulo, e o ganho
resultante é levado de volta ao tamanho da foto. Assim o fungo é corrigido exatamente onde está.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from .correct import CorrectionInfo, CorrectionParams, correct_image
from .mask import DefectMask
from .model import AnalysisParams

STEP = 4  # a grade de análise é 1/4 da resolução total: recortes alinhados a 4 px


class UnsupportedFrame(ValueError):
    pass


@dataclass(frozen=True)
class SensorRegion:
    x0: int
    y0: int
    w: int
    h: int

    @property
    def full(self) -> bool:
        return self.x0 == 0 and self.y0 == 0


def _align(v: float) -> int:
    return max(STEP, int(round(v / STEP)) * STEP)


def sensor_region(photo_hw: tuple[int, int], sensor_hw: tuple[int, int], digital_zoom: float = 1.0) -> SensorRegion:
    """Retângulo do sensor (resolução total) que a foto mostra."""
    ph, pw = photo_hw
    sh, sw = sensor_hw
    zoom = max(float(digital_zoom or 1.0), 1.0)
    if pw / ph >= sw / sh:  # mais larga ou igual ao sensor: ocupa a largura inteira
        w = sw / zoom
        h = w * ph / pw
    else:  # mais alta (ex.: 1:1): ocupa a altura inteira
        h = sh / zoom
        w = h * pw / ph
    w, h = _align(w), _align(h)
    if w > sw or h > sh:
        raise UnsupportedFrame(f"formato {pw}x{ph} não cabe no sensor {sw}x{sh}")
    return SensorRegion(_align((sw - w) / 2) if w < sw else 0, _align((sh - h) / 2) if h < sh else 0, w, h)


def crop_mask(dm: DefectMask, r: SensorRegion) -> DefectMask:
    """A máscara restrita ao retângulo ``r`` (mapas em resolução total e na grade de análise)."""
    if r.full and (r.h, r.w) == dm.shape:
        return dm
    q = dm.fungus_map.shape[0] / dm.shape[0]  # escala da grade de análise
    fy, fx = slice(r.y0, r.y0 + r.h), slice(r.x0, r.x0 + r.w)
    ay = slice(round(r.y0 * q), round(r.y0 * q) + round(r.h * q))
    ax = slice(round(r.x0 * q), round(r.x0 * q) + round(r.w * q))
    return replace(
        dm,
        mask=np.ascontiguousarray(dm.mask[fy, fx]),
        fungus_mask=np.ascontiguousarray(dm.fungus_mask[fy, fx]),
        hotpixel_mask=np.ascontiguousarray(dm.hotpixel_mask[fy, fx]),
        fungus_map=np.ascontiguousarray(dm.fungus_map[ay, ax]),
        sessions=np.ascontiguousarray(dm.sessions[ay, ax]),
    )


def correct_frame(
    img_bgr: np.ndarray,
    dm: DefectMask,
    params: CorrectionParams,
    analysis: AnalysisParams,
    fnumber: float | None = None,
    digital_zoom: float = 1.0,
) -> tuple[np.ndarray, CorrectionInfo | None, str]:
    """Corrige uma foto de qualquer formato/tamanho desta câmera. Retorna (imagem, info, descrição do quadro)."""
    ph, pw = img_bgr.shape[:2]
    r = sensor_region((ph, pw), dm.shape, digital_zoom)
    sub = crop_mask(dm, r)
    if (r.h, r.w) == (ph, pw):  # já está na escala do sensor (resolução total, com ou sem recorte)
        out, info = correct_image(img_bgr, sub, params, analysis, fnumber)
        return out, info, _describe(r, dm.shape, digital_zoom, resampled=False)
    interp = cv2.INTER_AREA if r.w < pw else cv2.INTER_CUBIC
    scaled = cv2.resize(img_bgr, (r.w, r.h), interpolation=interp)
    fixed, info = correct_image(scaled, sub, params, analysis, fnumber)
    # Leva o ganho (não a imagem) de volta ao tamanho da foto: os detalhes originais não são reamostrados.
    gain = (fixed.astype(np.float32) + 1.0) / (scaled.astype(np.float32) + 1.0)
    gain = cv2.resize(gain, (pw, ph), interpolation=cv2.INTER_LINEAR if r.w < pw else cv2.INTER_AREA)
    out = np.clip((img_bgr.astype(np.float32) + 1.0) * gain - 1.0 + 0.5, 0, 255).astype(np.uint8)
    return out, info, _describe(r, dm.shape, digital_zoom, resampled=True)


def _describe(r: SensorRegion, sensor_hw: tuple[int, int], zoom: float, resampled: bool) -> str:
    parts = []
    if zoom and zoom > 1.0:
        parts.append(f"zoom digital {zoom:.2f}x")
    if (r.h, r.w) != sensor_hw and not (zoom and zoom > 1.0):
        parts.append(f"recorte {r.w}x{r.h} do sensor")
    if resampled:
        parts.append("reamostrada")
    return ", ".join(parts) or "quadro inteiro"
