"""Leitura/escrita de imagens preservando EXIF.

As imagens são sempre lidas na orientação *do sensor* (ignorando a tag EXIF
Orientation). O defeito está fixo em coordenadas do sensor, então girar a
foto antes de aplicar a máscara a desalinharia. Como a tag Orientation é
preservada na saída, os visualizadores continuam exibindo a foto na posição
correta.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"})
JPEG_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg"})


class ImageReadError(Exception):
    """Arquivo ausente, corrompido ou em formato não suportado."""


def list_images(directory: Path) -> list[Path]:
    """Lista (ordenada) das imagens suportadas em ``directory``, sem recursão."""
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def read_image(path: Path) -> np.ndarray:
    """Lê uma imagem BGR 8 bits na orientação do sensor.

    Usa ``np.fromfile`` + ``cv2.imdecode`` para suportar caminhos com acentos no Windows.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise ImageReadError(f"não foi possível ler {path}: {exc}") from exc
    img = cv2.imdecode(data, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise ImageReadError(f"arquivo não é uma imagem válida: {path}")
    return img


def _exif_bytes(src: Path) -> bytes | None:
    """Bloco EXIF bruto da origem, copiado byte a byte.

    Não reserializar é intencional: o MakerNote da Canon (e de outros
    fabricantes) guarda offsets que quebram se o EXIF for remontado.
    Consequência: a miniatura embutida (160 px) continua sendo a original.
    """
    with Image.open(src) as pil:
        return pil.info.get("exif") or None


def _icc_profile(src: Path) -> bytes | None:
    with Image.open(src) as pil:
        return pil.info.get("icc_profile")


def write_image(dst: Path, img_bgr: np.ndarray, src: Path, jpeg_quality: int = 95) -> None:
    """Salva ``img_bgr`` em ``dst`` copiando EXIF e perfil ICC de ``src``.

    JPEG é codificado pelo Pillow com subamostragem 4:4:4 para minimizar perda
    de cor. Outros formatos são gravados pelo OpenCV, sem EXIF.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix.lower() in JPEG_EXTENSIONS:
        rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        kwargs: dict[str, object] = {"quality": jpeg_quality, "subsampling": 0, "optimize": True}
        if (exif := _exif_bytes(src)) is not None:
            kwargs["exif"] = exif
        if (icc := _icc_profile(src)) is not None:
            kwargs["icc_profile"] = icc
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", **kwargs)
        dst.write_bytes(buf.getvalue())
        return
    ok, buf = cv2.imencode(dst.suffix, img_bgr)
    if not ok:
        raise OSError(f"falha ao codificar {dst}")
    buf.tofile(str(dst))


def read_focal_length(path: Path) -> float | None:
    """Distância focal (mm) do EXIF, ou ``None`` se ausente/ilegível."""
    try:
        with Image.open(path) as pil:
            value = pil.getexif().get_ifd(ExifTags.IFD.Exif).get(ExifTags.Base.FocalLength)
        return float(value) if value else None
    except Exception:  # noqa: BLE001 - EXIF malformado não deve derrubar o lote
        return None
