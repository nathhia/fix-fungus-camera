"""Geração da máscara estática do defeito a partir de fotos de referência.

Dois tipos de defeito, com assinaturas diferentes:

* **Fungo** (filamentos no bloco óptico): sombra *semitransparente*, larga
  e desfocada, que escurece a imagem alguns por cento. Para isolá-la, cada
  referência vira um mapa ``log(imagem) - log(fundo local)`` calculado só
  nas regiões lisas (céu, parede). A mediana desses mapas entre várias
  fotos elimina o conteúdo das cenas e sobra o padrão fixo.
* **Pixels quentes** (o "ponto branco"): poucos px do sensor, muito
  brilhantes, na mesma posição em *todas* as fotos, sejam lisas ou não.
  São detectados contando em quantas referências o pixel se destaca da
  mediana local 7x7. Candidatos agrupados (ex.: o carimbo de data da
  câmera, que também é fixo) são descartados, porque pixel quente é isolado.

Saídas (em ``mask_dir``):

* ``mask.png`` - máscara binária final (defeito = 255) para ``cv2.inpaint``.
* ``fungus_mask.png`` / ``hotpixel_mask.png`` - as partes separadas.
* ``fungus_map.png`` - atenuação do fungo (PNG 16 bits, em log), usada pela
  correção *flat-field*.
* ``coverage.png`` - quantas referências lisas cobriram cada pixel.
* ``preview.jpg`` - a máscara sobreposta a uma referência, para inspeção.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .imageio import ImageReadError, list_images, read_focal_length, read_image

log = logging.getLogger(__name__)

# fungus_map.png guarda a atenuação (log) em 16 bits: valor = A / FUNGUS_MAP_MAX * 65535
FUNGUS_MAP_MAX: float = 1.0


@dataclass(frozen=True)
class MaskParams:
    """Parâmetros da extração. Distâncias em pixels da imagem *original*."""

    analysis_scale: float = 0.25  # resolução usada para o fungo (ele é suave, não precisa de 100%)
    background_sigma: float = 90.0  # escala do "fundo" local; deve ser >> largura dos filamentos
    flat_texture_max: float = 0.022  # textura fina máxima (log) para considerar uma região lisa
    fungus_threshold: float = 0.02  # atenuação (log ~ fração de luz perdida) que "semeia" um filamento
    fungus_threshold_low: float = 0.008  # histerese: pixels acima disto entram se conectados a uma semente
    seed_min_coverage: int = 3  # sementes só onde >= N referências lisas concordam (evita ruído)
    fungus_min_area: int = 2000  # remove manchas isoladas menores que isto (px²)
    fungus_dilate: int = 6  # margem extra ao redor dos filamentos
    fungus_smooth: float = 4.0  # suavização do mapa final (reduz ruído entre referências)
    hot_threshold: int = 12  # quanto o pixel quente sobressai da mediana 7x7 (0-255)
    hot_min_fraction: float = 0.7  # fração das referências em que o pixel precisa aparecer
    hot_max_area: int = 40  # componentes maiores não são pixels quentes
    hot_isolation: int = 60  # raio: >2 candidatos nessa vizinhança = texto/textura, não pixel quente
    hot_dilate: int = 3
    zoom_group_ratio: float = 1.25  # focais dentro desse fator formam um grupo (ex.: 5.0-6.25 mm)
    zoom_min_own_coverage: float = 0.15  # fração mínima do quadro coberta pelo próprio grupo


def _normalized_blur(values: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Blur gaussiano que ignora pixels com peso 0 (convolução normalizada)."""
    num = cv2.GaussianBlur(values * weight, (0, 0), sigma)
    den = cv2.GaussianBlur(weight, (0, 0), sigma)
    return num / np.maximum(den, 1e-6)


def fungus_residual(img_bgr: np.ndarray, p: MaskParams) -> np.ndarray:
    """Mapa ``log(I) - log(fundo)`` na escala de análise; NaN onde a foto não é lisa."""
    small = cv2.resize(img_bgr, None, fx=p.analysis_scale, fy=p.analysis_scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    logi = np.log(gray + 1.0)

    # Textura em escala menor que a largura dos filamentos: céu/parede ~ 0, folhagem >> 0.
    fine = logi - cv2.GaussianBlur(logi, (0, 0), 1.5)
    texture = np.sqrt(cv2.GaussianBlur(fine * fine, (0, 0), 6))
    flat = (texture < p.flat_texture_max) & (gray > 30) & (gray < 250)
    flat = cv2.erode(flat.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)

    sigma = p.background_sigma * p.analysis_scale
    bg = _normalized_blur(logi, flat, sigma)
    # 2ª passada sem os próprios filamentos, para que eles não puxem o fundo para baixo.
    bg = _normalized_blur(logi, flat * ((logi - bg) > -p.fungus_threshold), sigma)

    residual = logi - bg
    residual[flat == 0] = np.nan
    return residual


def hot_pixel_hits(img_bgr: np.ndarray, threshold: int) -> np.ndarray:
    """Máscara (bool, resolução total) de pixels muito mais brilhantes que a vizinhança 7x7."""
    local = cv2.medianBlur(img_bgr, 7)
    diff = cv2.subtract(img_bgr, local).max(axis=2)
    return diff > threshold


def _clean_components(mask: np.ndarray, min_area: int = 0, max_area: int | None = None) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA]
    keep = area >= min_area
    if max_area is not None:
        keep &= area <= max_area
    keep[0] = False
    return keep[labels]


def _hysteresis(values: np.ndarray, low: float, high: float, seed_mask: np.ndarray | None = None) -> np.ndarray:
    """Regiões acima de ``low`` que contêm ao menos um pixel acima de ``high`` (dentro de ``seed_mask``)."""
    n, labels = cv2.connectedComponents((values > low).astype(np.uint8), connectivity=8)
    seeds = values > high
    if seed_mask is not None:
        seeds &= seed_mask
    seeded = np.zeros(n, bool)
    seeded[np.unique(labels[seeds])] = True
    seeded[0] = False
    return seeded[labels]


def _isolated_components(mask: np.ndarray, radius: int, max_neighbors: int = 2) -> np.ndarray:
    """Mantém só componentes com no máximo ``max_neighbors`` componentes (incl. ele) num raio ``radius``."""
    m = mask.astype(np.uint8)
    n, labels, _, cents = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return mask
    _, groups = cv2.connectedComponents(cv2.dilate(m, _disk(radius // 2)))
    gid = groups[cents[1:, 1].astype(int), cents[1:, 0].astype(int)]
    per_group = np.bincount(gid)
    keep = np.zeros(n, bool)
    keep[1:] = per_group[gid] <= max_neighbors
    return keep[labels]


def _disk(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


@dataclass
class DefectMask:
    mask: np.ndarray  # uint8 {0,255}, resolução total: união fungo + pixels quentes
    fungus_mask: np.ndarray  # uint8 {0,255}
    hotpixel_mask: np.ndarray  # uint8 {0,255}
    fungus_map: np.ndarray  # float32 >= 0, atenuação em log (0 = sem fungo)
    coverage: np.ndarray  # uint8, nº de referências lisas por pixel (escala de análise)

    @property
    def shape(self) -> tuple[int, int]:
        return self.mask.shape[:2]

    def save(self, directory: Path, preview_base: np.ndarray | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(directory / "mask.png"), self.mask)
        cv2.imwrite(str(directory / "fungus_mask.png"), self.fungus_mask)
        cv2.imwrite(str(directory / "hotpixel_mask.png"), self.hotpixel_mask)
        fmap16 = np.clip(self.fungus_map / FUNGUS_MAP_MAX * 65535, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(directory / "fungus_map.png"), fmap16)
        cv2.imwrite(str(directory / "coverage.png"), self.coverage)
        if preview_base is not None:
            cv2.imwrite(str(directory / "preview.jpg"), make_preview(preview_base, self))

    @classmethod
    def load(cls, directory: Path) -> "DefectMask":
        def _read(name: str, flags: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray:
            img = cv2.imread(str(directory / name), flags)
            if img is None:
                raise FileNotFoundError(f"{directory / name} não encontrado; rode 'build-mask' primeiro")
            return img

        fmap = _read("fungus_map.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535 * FUNGUS_MAP_MAX
        coverage = cv2.imread(str(directory / "coverage.png"), cv2.IMREAD_GRAYSCALE)
        return cls(
            mask=_read("mask.png"),
            fungus_mask=_read("fungus_mask.png"),
            hotpixel_mask=_read("hotpixel_mask.png"),
            fungus_map=fmap,
            coverage=coverage if coverage is not None else np.zeros((1, 1), np.uint8),
        )


def make_preview(base_bgr: np.ndarray, dm: DefectMask, max_side: int = 2000) -> np.ndarray:
    """Referência com fungo em magenta e pixels quentes circulados em ciano."""
    out = base_bgr.copy()
    fm = dm.fungus_mask > 0
    out[fm] = (0.5 * out[fm] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
    n, _, _, cents = cv2.connectedComponentsWithStats(dm.hotpixel_mask)
    for i in range(1, n):
        cv2.circle(out, tuple(int(v) for v in cents[i]), 40, (255, 255, 0), 6)
    s = max_side / max(out.shape[:2])
    return cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else out


def _median_residual(residuals: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mediana ignorando NaN + nº de amostras válidas por pixel."""
    stack = np.stack(residuals)
    coverage = np.sum(~np.isnan(stack), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # fatias só-NaN: tratadas via ``coverage``
        med = np.nanmedian(stack, axis=0)
    return med, coverage


def _fungus_from_median(
    med: np.ndarray, coverage: np.ndarray, shape: tuple[int, int], p: MaskParams
) -> tuple[np.ndarray, np.ndarray]:
    """Mediana dos resíduos (escala de análise) -> (máscara bool, mapa de atenuação) em resolução total."""
    h, w = shape
    holes = coverage == 0
    med = np.nan_to_num(med, nan=0.0).astype(np.float32)
    if holes.any():
        # Regiões que nenhuma referência lisa cobriu: interpola o mapa a partir das bordas.
        log.warning("%.1f%% do quadro sem referência lisa; valores interpolados", 100 * holes.mean())
        med = cv2.inpaint(med, holes.astype(np.uint8), 15, cv2.INPAINT_TELEA)
    med = cv2.GaussianBlur(med, (0, 0), p.fungus_smooth * p.analysis_scale)

    atten = cv2.resize(np.clip(-med, 0, None), (w, h), interpolation=cv2.INTER_CUBIC).clip(0, None)
    reliable = cv2.resize((coverage >= p.seed_min_coverage).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    fungus = _hysteresis(atten, p.fungus_threshold_low, p.fungus_threshold, seed_mask=reliable > 0)
    fungus = _clean_components(fungus, min_area=p.fungus_min_area)
    fungus = cv2.dilate(fungus.astype(np.uint8), _disk(p.fungus_dilate)) > 0

    # Mapa de atenuação restrito à máscara, com borda suave para não criar degraus.
    feather = cv2.GaussianBlur(fungus.astype(np.float32), (0, 0), max(1.0, p.fungus_dilate / 2))
    return fungus, (atten * feather).astype(np.float32)


def _assemble(fungus: np.ndarray, fungus_map: np.ndarray, hot: np.ndarray, coverage: np.ndarray) -> DefectMask:
    u8 = lambda m: m.astype(np.uint8) * 255  # noqa: E731
    return DefectMask(
        mask=u8(fungus | hot),
        fungus_mask=u8(fungus),
        hotpixel_mask=u8(hot),
        fungus_map=fungus_map,
        coverage=np.clip(coverage, 0, 255).astype(np.uint8),
    )


def _zoom_groups(focals: list[float | None], ratio: float) -> list[list[int]]:
    """Agrupa índices por distância focal: um grupo abrange no máximo ``ratio`` (ex. 5.0-6.25 mm)."""
    known = sorted((f, i) for i, f in enumerate(focals) if f is not None)
    groups: list[list[int]] = []
    start = 0.0
    for f, i in known:
        if not groups or f > start * ratio:
            groups.append([])
            start = f
        groups[-1].append(i)
    return groups


@dataclass
class MaskSet:
    """Máscara global + máscaras específicas por faixa de zoom.

    O fungo não está exatamente sobre o sensor: com o zoom, sua sombra muda um
    pouco de posição e de nitidez. Cada grupo de zoom usa as próprias
    referências onde há cobertura suficiente e completa o resto com o mapa global.
    """

    global_mask: DefectMask
    groups: dict[float, DefectMask]  # distância focal representativa (mm) -> máscara
    group_ratio: float = 1.25

    def select(self, focal: float | None) -> tuple[DefectMask, float | None]:
        """Máscara para uma foto com a distância focal dada (e a focal do grupo escolhido)."""
        if focal is None or not self.groups:
            return self.global_mask, None
        best = min(self.groups, key=lambda g: abs(np.log(focal / g)))
        if abs(np.log(focal / best)) <= np.log(self.group_ratio):
            return self.groups[best], best
        return self.global_mask, None

    def save(self, directory: Path, preview_base: np.ndarray | None = None) -> None:
        self.global_mask.save(directory, preview_base)
        subdirs: dict[str, float] = {}
        for f, dm in self.groups.items():
            sub = f"zoom_{f:.1f}mm"
            dm.save(directory / sub)
            subdirs[sub] = f
        manifest = {"group_ratio": self.group_ratio, "groups": subdirs}
        (directory / "zoom_groups.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, directory: Path) -> "MaskSet":
        groups: dict[float, DefectMask] = {}
        ratio = 1.25
        manifest_path = directory / "zoom_groups.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            ratio = float(manifest.get("group_ratio", ratio))
            groups = {float(f): DefectMask.load(directory / sub) for sub, f in manifest["groups"].items()}
        return cls(global_mask=DefectMask.load(directory), groups=groups, group_ratio=ratio)


def build_mask(reference_dir: Path, params: MaskParams | None = None) -> tuple[MaskSet, np.ndarray]:
    """Gera as máscaras a partir de todas as imagens em ``reference_dir``.

    Retorna ``(máscaras, referência mais lisa)``; a segunda serve para o preview.
    Imagens ilegíveis ou com resolução diferente da primeira são ignoradas.
    """
    p = params or MaskParams()
    paths = list_images(reference_dir)
    if not paths:
        raise FileNotFoundError(f"nenhuma imagem em {reference_dir}")

    shape: tuple[int, int] | None = None
    residuals: list[np.ndarray] = []
    focals: list[float | None] = []
    hot_count: np.ndarray | None = None
    best_flat, best_img = -1.0, None

    for path in tqdm(paths, desc="Analisando referências", unit="img"):
        try:
            img = read_image(path)
        except ImageReadError as exc:
            log.warning("%s", exc)
            continue
        if shape is None:
            shape = img.shape[:2]
            hot_count = np.zeros(shape, np.uint16)
        if img.shape[:2] != shape:
            log.warning("%s ignorada: resolução %s != %s", path.name, img.shape[:2], shape)
            continue
        assert hot_count is not None
        hot_count += hot_pixel_hits(img, p.hot_threshold)
        r = fungus_residual(img, p)
        residuals.append(r)
        focals.append(read_focal_length(path))
        flat_frac = float(np.mean(~np.isnan(r)))
        log.info("%s: %.0f%% de área lisa, f=%s mm", path.name, 100 * flat_frac, focals[-1])
        if flat_frac > best_flat:
            best_flat, best_img = flat_frac, img

    if not residuals or shape is None or hot_count is None or best_img is None:
        raise ValueError("nenhuma referência válida")

    # --- pixels quentes (do sensor: iguais em qualquer zoom) -----------------
    hot = hot_count >= max(1, int(np.ceil(p.hot_min_fraction * len(residuals))))
    hot = _clean_components(hot, max_area=p.hot_max_area)
    hot = _isolated_components(hot, p.hot_isolation)
    hot = cv2.dilate(hot.astype(np.uint8), _disk(p.hot_dilate)) > 0
    log.info("pixels quentes encontrados: %d", cv2.connectedComponents(hot.astype(np.uint8))[0] - 1)

    # --- fungo: mapa global ---------------------------------------------------
    g_med, g_cov = _median_residual(residuals)
    fungus, fmap = _fungus_from_median(g_med, g_cov, shape, p)
    global_mask = _assemble(fungus, fmap, hot, g_cov)

    # --- fungo: um mapa por faixa de zoom -------------------------------------
    groups: dict[float, DefectMask] = {}
    for idx in _zoom_groups(focals, p.zoom_group_ratio):
        med, cov = _median_residual([residuals[i] for i in idx])
        own = cov >= p.seed_min_coverage
        if own.mean() < p.zoom_min_own_coverage:
            continue  # poucas referências lisas nesse zoom: o mapa global serve melhor
        med = np.where(own, med, g_med)
        cov = np.where(own, cov, g_cov)
        focal = float(np.median([focals[i] for i in idx]))  # type: ignore[type-var]
        log.info("grupo de zoom %.1f mm: %d refs, %.0f%% do quadro com mapa próprio", focal, len(idx), 100 * own.mean())
        fungus, fmap = _fungus_from_median(med, cov, shape, p)
        groups[focal] = _assemble(fungus, fmap, hot, cov)

    return MaskSet(global_mask, groups, p.zoom_group_ratio), best_img
