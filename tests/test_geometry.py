from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from fungusfix.correct import CorrectionParams
from fungusfix.geometry import correct_frame, crop_mask, sensor_region
from fungusfix.imageio import read_digital_zoom, read_image, write_image
from fungusfix.mask import DefectMask
from fungusfix.model import AnalysisParams, blur_map

SENSOR = (3456, 4608)


@pytest.mark.parametrize(
    ("photo", "zoom", "expected"),
    [
        ((3456, 4608), 1.0, (0, 0, 4608, 3456)),  # L 4:3
        ((2592, 4608), 1.0, (0, 432, 4608, 2592)),  # 16:9
        ((3072, 4608), 1.0, (0, 192, 4608, 3072)),  # 3:2
        ((3456, 3456), 1.0, (576, 0, 3456, 3456)),  # 1:1
        ((480, 640), 1.0, (0, 0, 4608, 3456)),  # S: sensor inteiro reduzido
        ((3456, 4608), 2.0, (1152, 864, 2304, 1728)),  # zoom digital 2x
    ],
)
def test_sensor_region(photo, zoom, expected):
    r = sensor_region(photo, SENSOR, zoom)
    assert (r.x0, r.y0, r.w, r.h) == expected
    assert r.x0 % 4 == r.y0 % 4 == r.w % 4 == r.h % 4 == 0


def _mask(shape=(864, 1152), q=0.25) -> DefectMask:
    """Máscara pequena com uma 'teia' sintética (linhas) e uma mancha."""
    h, w = shape
    fmap = np.zeros((int(h * q), int(w * q), 3), np.float32)
    for x in range(20, fmap.shape[1] - 20, 23):
        cv2.line(fmap, (x, 10), (x + 30, fmap.shape[0] - 10), (0.08, 0.08, 0.08), 1)
    cv2.circle(fmap, (fmap.shape[1] // 2, fmap.shape[0] // 2), 12, (0.05, 0.06, 0.07), -1)
    binary = (cv2.resize(fmap.mean(2), (w, h)) > 0.01).astype(np.uint8) * 255
    return DefectMask(
        mask=binary,
        fungus_mask=binary.copy(),
        hotpixel_mask=np.zeros((h, w), np.uint8),
        fungus_map=fmap,
        sessions=(fmap.mean(2) > 0).astype(np.uint8) * 3,
        prior_sigma=1.0,
        prior_k=1.0,
    )


def test_crop_mask_keeps_grids_aligned():
    dm = _mask()
    r = sensor_region((486, 864), dm.shape)  # 16:9
    sub = crop_mask(dm, r)
    assert sub.shape == (r.h, r.w)
    assert sub.fungus_map.shape[:2] == (r.h // 4, r.w // 4)
    assert np.array_equal(sub.fungus_map, dm.fungus_map[r.y0 // 4 : (r.y0 + r.h) // 4, r.x0 // 4 : (r.x0 + r.w) // 4])


def _shadowed(dm: DefectMask, sigma=1.0, k=1.0) -> tuple[np.ndarray, np.ndarray]:
    """Cena lisa com ruído leve e a sombra do mapa aplicada: (com sombra, sem sombra)."""
    h, w = dm.shape
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    clean = np.dstack([140 + 40 * xx / w, 150 + 30 * yy / h, np.full((h, w), 160.0)]) + rng.normal(0, 1.5, (h, w, 3))
    att = cv2.resize(np.exp(-k * blur_map(dm.fungus_map, sigma)), (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(clean * att + 0.5, 0, 255).astype(np.uint8), np.clip(clean + 0.5, 0, 255).astype(np.uint8)


def _leftover(fixed: np.ndarray, before: np.ndarray, clean: np.ndarray, where: np.ndarray) -> float:
    def lg(x: np.ndarray) -> np.ndarray:
        return np.log(x.astype(np.float32) + 1.0)

    return float(np.abs(lg(fixed) - lg(clean))[where].mean() / np.abs(lg(before) - lg(clean))[where].mean())


@pytest.mark.parametrize(
    ("photo", "zoom"),
    [((864, 1152), 1.0), ((648, 1152), 1.0), ((864, 864), 1.0), ((432, 576), 1.0), ((864, 1152), 2.0)],
    ids=["inteira", "16:9", "1:1", "reduzida", "zoom 2x"],
)
def test_correct_frame_removes_shadow_in_any_format(photo, zoom):
    dm = _mask()
    shadowed, clean = _shadowed(dm)
    r = sensor_region(photo, dm.shape, zoom)

    def take(x: np.ndarray) -> np.ndarray:
        crop = x[r.y0 : r.y0 + r.h, r.x0 : r.x0 + r.w]
        return crop if crop.shape[:2] == photo else cv2.resize(crop, (photo[1], photo[0]), interpolation=cv2.INTER_AREA)

    before, truth = take(shadowed), take(clean)
    fixed, info, _ = correct_frame(before, dm, CorrectionParams(), AnalysisParams(), None, zoom)
    assert fixed.shape == before.shape
    assert info is not None
    left = _leftover(fixed, before, truth, take(dm.fungus_mask) > 0)
    assert left < 0.5


def test_read_digital_zoom(tmp_path: Path):
    src = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8)).save(src)
    assert read_digital_zoom(src) == 1.0
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0xA404] = 2.5  # DigitalZoomRatio
    Image.new("RGB", (8, 8)).save(src, exif=exif)
    assert read_digital_zoom(src) == pytest.approx(2.5)


def test_write_image_keeps_exif(tmp_path: Path):
    src = tmp_path / "src.jpg"
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0x829D] = 5.6  # FNumber
    Image.new("RGB", (16, 16), (90, 120, 150)).save(src, exif=exif)
    dst = tmp_path / "dst.jpg"
    write_image(dst, read_image(src), src)
    assert Image.open(dst).getexif().get_ifd(0x8769)[0x829D] == pytest.approx(5.6)
