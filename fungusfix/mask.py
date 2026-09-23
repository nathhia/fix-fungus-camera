"""Geração da máscara estática do defeito a partir de fotos de referência.

Dois tipos de defeito, com assinaturas diferentes:

* **Fungo / manchas no bloco óptico**: sombra *semitransparente* e
  desfocada, que escurece a imagem alguns por cento, às vezes com cor (a
  mancha marrom absorve mais azul). Cada referência vira um mapa
  ``log(imagem) - log(fundo local)`` por canal, calculado só nas regiões
  lisas (céu, parede). O mapa final é a mediana entre **sessões**
  independentes (fotos agrupadas pela hora da captura): um horizonte que
  aparece na mesma altura em 5 fotos da praia não se repete no jantar, e o
  fungo sim.
* **Pixels quentes** (o "ponto branco"): poucos px do sensor, muito
  brilhantes, na mesma posição em *todas* as fotos. São detectados
  contando em quantas referências o pixel se destaca da mediana 7x7.
  Candidatos agrupados (ex.: o carimbo de data, também fixo) são descartados.

Saídas (em ``mask_dir``):

* ``mask.png`` - máscara binária final (defeito = 255), usada por ``--method inpaint``.
* ``fungus_mask.png`` / ``hotpixel_mask.png`` - as partes separadas.
* ``fungus_map.png`` - atenuação por canal (BGR, PNG 16 bits, escala de análise).
* ``sessions.png`` - quantas sessões independentes confirmaram cada ponto.
* ``meta.json`` - parâmetros e o desfoque/intensidade típicos da sombra.
* ``preview.jpg`` - a máscara sobreposta a uma referência, para inspeção.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .imageio import ImageReadError, list_images, read_capture_time, read_fnumber, read_focal_length, read_image
from .model import AnalysisParams, ApertureEntry, fit_shadow, residual

log = logging.getLogger(__name__)

# fungus_map.png guarda a atenuação (log) em 16 bits: valor = A / FUNGUS_MAP_MAX * 65535
FUNGUS_MAP_MAX: float = 1.0


@dataclass(frozen=True)
class MaskParams:
    """Parâmetros da extração. Distâncias de hot pixels em px da imagem original."""

    analysis: AnalysisParams = field(default_factory=lambda: AnalysisParams(background_sigma=120.0))
    fungus_threshold: float = 0.02  # atenuação (log ~ fração de luz perdida) que "semeia" um defeito
    fungus_threshold_low: float = 0.008  # histerese: pixels acima disto entram se conectados a uma semente
    seed_min_sessions: int = 2  # sementes só onde >= N sessões independentes concordam
    min_images: int = 2  # e onde >= N fotos lisas cobriram o ponto
    fungus_min_area: int = 125  # px de análise (~2000 px na resolução original)
    single_session_weight: float = 0.5  # peso do mapa onde só uma sessão tem área lisa
    session_gap_minutes: float = 20.0  # fotos com intervalo maior formam outra sessão
    hot_threshold: int = 12  # quanto o pixel quente sobressai da mediana 7x7 (0-255)
    hot_min_fraction: float = 0.7  # fração das referências em que o pixel precisa aparecer
    hot_max_area: int = 40  # componentes maiores não são pixels quentes
    hot_isolation: int = 60  # raio: >2 candidatos nessa vizinhança = texto/textura, não pixel quente
    hot_dilate: int = 3
    zoom_group_ratio: float = 1.5  # focais dentro desse fator formam um grupo (ex.: 5.0-7.5 mm)
    zoom_min_own_coverage: float = 0.15  # fração mínima do quadro coberta pelo próprio grupo
    sharp_fnumber_margin: float = 0.6  # calibração: usa as fotos com F até isto abaixo do maior F do grupo
    default_prior_sigma: float = 2.0  # σ/k típicos quando não há foto real do zoom para medir
    default_prior_k: float = 0.8


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


def _nanmedian(stack: np.ndarray, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # fatias só-NaN viram NaN, tratadas pelo chamador
        return np.nanmedian(stack, axis=axis)


@dataclass
class DefectMask:
    mask: np.ndarray  # uint8 {0,255}, resolução total: união fungo + pixels quentes
    fungus_mask: np.ndarray  # uint8 {0,255}, resolução total
    hotpixel_mask: np.ndarray  # uint8 {0,255}, resolução total
    fungus_map: np.ndarray  # float32 (h, w, 3) >= 0 na escala de análise: atenuação log por canal BGR
    sessions: np.ndarray  # uint8 (h, w): nº de sessões independentes que confirmaram cada ponto
    prior_sigma: float = 0.0  # desfoque típico da sombra (px de análise), para fotos sem área lisa
    prior_k: float = 1.0  # intensidade típica
    aperture_table: list[ApertureEntry] = field(default_factory=list)  # σ/k medidos por abertura (calibração)

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
        cv2.imwrite(str(directory / "sessions.png"), self.sessions)
        meta = {
            "prior_sigma": self.prior_sigma,
            "prior_k": self.prior_k,
            "aperture_table": [asdict(e) for e in self.aperture_table],
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))
        if preview_base is not None:
            cv2.imwrite(str(directory / "preview.jpg"), make_preview(preview_base, self))

    @classmethod
    def load(cls, directory: Path) -> "DefectMask":
        def _read(name: str, flags: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray:
            img = cv2.imread(str(directory / name), flags)
            if img is None:
                raise FileNotFoundError(f"{directory / name} não encontrado; rode 'build-mask' primeiro")
            return img

        meta_path = directory / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} não encontrado; máscara de versão antiga, rode 'build-mask' de novo")
        meta = json.loads(meta_path.read_text())
        fmap = _read("fungus_map.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535 * FUNGUS_MAP_MAX
        return cls(
            mask=_read("mask.png"),
            fungus_mask=_read("fungus_mask.png"),
            hotpixel_mask=_read("hotpixel_mask.png"),
            fungus_map=fmap,
            sessions=_read("sessions.png"),
            prior_sigma=float(meta["prior_sigma"]),
            prior_k=float(meta["prior_k"]),
            aperture_table=[ApertureEntry(**e) for e in meta.get("aperture_table", [])],
        )


def make_preview(base_bgr: np.ndarray, dm: DefectMask, max_side: int = 2000) -> np.ndarray:
    """Referência com o defeito em magenta (intensidade = atenuação) e pixels quentes circulados em ciano."""
    h, w = base_bgr.shape[:2]
    a = cv2.resize(dm.fungus_map.mean(axis=2), (w, h), interpolation=cv2.INTER_LINEAR)
    alpha = np.clip(a / 0.08, 0, 0.8)[..., None]
    out = (base_bgr * (1 - alpha) + np.array([255, 0, 255]) * alpha).astype(np.uint8)
    n, _, _, cents = cv2.connectedComponentsWithStats(dm.hotpixel_mask)
    for i in range(1, n):
        cv2.circle(out, tuple(int(v) for v in cents[i]), 40, (255, 255, 0), 6)
    s = max_side / max(out.shape[:2])
    return cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else out


@dataclass
class _Reference:
    name: str
    residual: np.ndarray  # (h, w, 3) escala de análise, NaN onde não é liso
    focal: float | None
    time: datetime | None
    fnumber: float | None = None
    session: int = -1


def _assign_sessions(refs: list[_Reference], gap_minutes: float) -> int:
    """Agrupa por horário: intervalo > ``gap_minutes`` inicia outra sessão. Sem horário = sessão própria."""
    timed = sorted((r for r in refs if r.time is not None), key=lambda r: r.time)  # type: ignore[arg-type,return-value]
    session, last = -1, None
    for r in timed:
        if last is None or r.time - last > timedelta(minutes=gap_minutes):  # type: ignore[operator]
            session += 1
        r.session, last = session, r.time
    for r in refs:
        if r.time is None:
            session += 1
            r.session = session
    return session + 1


def _defect_map(refs: list[_Reference], shape: tuple[int, int], hot: np.ndarray, p: MaskParams) -> DefectMask:
    """Mapa de atenuação a partir de um conjunto de referências (todas do mesmo zoom)."""
    h, w = shape
    per_session: list[np.ndarray] = []
    images = np.zeros(refs[0].residual.shape[:2], np.int32)
    for s in sorted({r.session for r in refs}):
        stack = np.stack([r.residual for r in refs if r.session == s])
        images += np.sum(~np.isnan(stack[..., 0]), axis=0)
        per_session.append(_nanmedian(stack))
    sess = np.stack(per_session)
    n_sessions = np.sum(~np.isnan(sess[..., 0]), axis=0)

    # O defeito precisa escurecer a imagem em *todas* as sessões que cobrem o ponto: usa o mínimo
    # entre elas. Onde só uma sessão cobre, não há como separar cena de defeito: entra pela metade.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        weakest = np.nanmin(-sess, axis=0)
    single_weight = 1.0 if len(per_session) == 1 else p.single_session_weight
    a = np.where(n_sessions[..., None] >= 2, weakest, weakest * single_weight)
    a = cv2.GaussianBlur(np.nan_to_num(a, nan=0.0).astype(np.float32), (0, 0), 1.0)
    # Com poucas referências (ex.: um zoom usado numa sessão só), exige o que houver.
    min_sessions = min(p.seed_min_sessions, len(per_session))
    useful = sum(1 for r in refs if np.mean(~np.isnan(r.residual[..., 0])) > 0.01)
    min_images = min(p.min_images, max(1, useful))
    confirmed = (n_sessions >= min_sessions) & (images >= min_images)
    return _finalize(a, confirmed, images >= min_images, refs, shape, hot, n_sessions, p)


def _finalize(
    a: np.ndarray,
    seeds_allowed: np.ndarray,
    allowed: np.ndarray,
    prior_refs: list[_Reference],
    shape: tuple[int, int],
    hot: np.ndarray,
    confirmations: np.ndarray,
    p: MaskParams,
) -> DefectMask:
    """Atenuação bruta ``a`` (escala de análise) -> máscara final, com suporte por histerese e σ/k típicos."""
    h, w = shape
    lum = a.mean(axis=2)
    support = _hysteresis(lum, p.fungus_threshold_low, p.fungus_threshold, seed_mask=seeds_allowed)
    support &= allowed
    support = _clean_components(support, min_area=p.fungus_min_area)
    support = cv2.dilate(support.astype(np.uint8), _disk(1)) > 0
    feather = cv2.GaussianBlur(support.astype(np.float32), (0, 0), 1.5)
    fmap = (np.clip(a, 0, None) * feather[..., None]).astype(np.float32)

    # Desfoque/intensidade típicos em fotos reais: usados quando a foto não tem área lisa para medir.
    fits = [f for f in (fit_shadow(r.residual, fmap, p.analysis) for r in prior_refs) if f.reliable]
    prior_sigma = float(np.median([f.sigma for f in fits])) if fits else p.default_prior_sigma
    prior_k = float(np.median([f.k for f in fits])) if fits else p.default_prior_k

    fungus = cv2.resize((fmap.mean(axis=2) > p.fungus_threshold_low).astype(np.uint8), (w, h), cv2.INTER_NEAREST)
    fungus = cv2.dilate(fungus, _disk(4)) > 0
    u8 = lambda m: m.astype(np.uint8) * 255  # noqa: E731
    return DefectMask(
        mask=u8(fungus | hot),
        fungus_mask=u8(fungus),
        hotpixel_mask=u8(hot),
        fungus_map=fmap,
        sessions=np.clip(confirmations, 0, 255).astype(np.uint8),
        prior_sigma=prior_sigma,
        prior_k=prior_k,
    )


def _calibration_map(
    members: list[_Reference],
    prior_refs: list[_Reference],
    shape: tuple[int, int],
    hot: np.ndarray,
    p: MaskParams,
) -> DefectMask:
    """Mapa a partir de fotos de calibração (campo branco desfocado) de um mesmo zoom.

    Usa as fotos de maior F (sombra mais nítida): desfocar um mapa nítido para
    casar com uma foto em F/3.5 funciona; o contrário não.
    """
    fmax = max((r.fnumber or 0.0) for r in members)
    sharp = [r for r in members if (r.fnumber or 0.0) >= fmax - p.sharp_fnumber_margin]
    if len(sharp) < 2:
        sharp = members
    stack = np.stack([r.residual for r in sharp])
    images = np.sum(~np.isnan(stack[..., 0]), axis=0)
    a = np.nan_to_num(-_nanmedian(stack), nan=0.0).astype(np.float32)
    # Onde nenhuma foto é lisa (ex.: o carimbo de data, sempre no mesmo lugar) não há como medir:
    # a região fica sem correção em vez de receber um valor inventado.
    covered = images > 0
    a = cv2.GaussianBlur(a, (0, 0), 0.7)
    dm = _finalize(a, covered, covered, prior_refs, shape, hot, images, p)
    dm.aperture_table = _aperture_table(members, dm.fungus_map, p)
    return dm


def _aperture_table(members: list[_Reference], fmap: np.ndarray, p: MaskParams) -> list[ApertureEntry]:
    """Desfoque e intensidade da sombra por abertura, medidos nas próprias fotos de calibração."""
    by_f: dict[float, list[tuple[float, float]]] = {}
    for r in members:
        if r.fnumber is None:
            continue
        f = fit_shadow(r.residual, fmap, p.analysis)
        if f.t >= 20 and f.sigma < max(p.analysis.blur_sigmas):  # σ no limite = cena, não sombra
            by_f.setdefault(round(r.fnumber, 1), []).append((f.sigma, f.k))
    table = [
        ApertureEntry(fn, float(np.median([s for s, _ in v])), float(np.median([k for _, k in v])))
        for fn, v in sorted(by_f.items())
    ]
    log.info("abertura -> (desfoque, intensidade): %s", [(e.fnumber, e.sigma, round(e.k, 2)) for e in table])
    return table


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
    """Máscara global + máscaras por faixa de zoom (a sombra muda de posição com o zoom)."""

    global_mask: DefectMask
    groups: dict[float, DefectMask]  # distância focal representativa (mm) -> máscara
    group_ratio: float = 1.25
    calibrated: bool = False  # grupos vindos de fotos de calibração cobrem toda a faixa de zoom

    def select(self, focal: float | None) -> tuple[DefectMask, float | None]:
        """Máscara para uma foto com a distância focal dada (e a focal do grupo escolhido)."""
        if focal is None or not self.groups:
            return self.global_mask, None
        best = min(self.groups, key=lambda g: abs(np.log(focal / g)))
        if self.calibrated or abs(np.log(focal / best)) <= np.log(self.group_ratio):
            return self.groups[best], best
        return self.global_mask, None

    def save(self, directory: Path, preview_base: np.ndarray | None = None, params: MaskParams | None = None) -> None:
        self.global_mask.save(directory, preview_base)
        subdirs: dict[str, float] = {}
        for f, dm in self.groups.items():
            sub = f"zoom_{f:.1f}mm"
            dm.save(directory / sub)
            subdirs[sub] = f
        manifest = {
            "group_ratio": self.group_ratio,
            "calibrated": self.calibrated,
            "groups": subdirs,
            "analysis": asdict((params or MaskParams()).analysis),
        }
        (directory / "zoom_groups.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, directory: Path) -> tuple["MaskSet", AnalysisParams]:
        manifest = json.loads((directory / "zoom_groups.json").read_text())
        groups = {float(f): DefectMask.load(directory / sub) for sub, f in manifest["groups"].items()}
        a = manifest.get("analysis", {})
        analysis = AnalysisParams(**{**a, "blur_sigmas": tuple(a["blur_sigmas"])}) if a else AnalysisParams()
        masks = cls(
            DefectMask.load(directory), groups, float(manifest["group_ratio"]), manifest.get("calibrated", False)
        )
        return masks, analysis


def build_mask(
    reference_dir: Path, params: MaskParams | None = None, calibration_dir: Path | None = None
) -> tuple[MaskSet, np.ndarray]:
    """Gera as máscaras a partir das fotos em ``reference_dir`` (e ``calibration_dir``, se houver).

    * ``reference_dir``: fotos reais em resolução total. Definem a resolução,
      os pixels quentes e o desfoque/intensidade típicos da sombra.
    * ``calibration_dir``: fotos de um campo branco desfocado, em vários
      zooms e aberturas (podem ter resolução menor). Quando existem, o mapa
      do fungo vem delas, por faixa de zoom.

    Retorna ``(máscaras, referência mais lisa)``; a segunda serve para o preview.
    Imagens ilegíveis ou com resolução diferente da primeira são ignoradas.
    """
    p = params or MaskParams()
    paths = list_images(reference_dir)
    if not paths:
        raise FileNotFoundError(f"nenhuma imagem em {reference_dir}")

    shape: tuple[int, int] | None = None
    refs: list[_Reference] = []
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
        r = residual(img, p.analysis)
        refs.append(_Reference(path.name, r, read_focal_length(path), read_capture_time(path), read_fnumber(path)))
        flat_frac = float(np.mean(~np.isnan(r[..., 0])))
        log.info("%s: %.0f%% de área lisa, f=%s mm", path.name, 100 * flat_frac, refs[-1].focal)
        if flat_frac > best_flat:
            best_flat, best_img = flat_frac, img

    if not refs or shape is None or hot_count is None or best_img is None:
        raise ValueError("nenhuma referência válida")
    n_sessions = _assign_sessions(refs, p.session_gap_minutes)
    log.info("%d referências em %d sessões independentes", len(refs), n_sessions)

    # --- pixels quentes (do sensor: iguais em qualquer zoom) -----------------
    hot = hot_count >= max(1, int(np.ceil(p.hot_min_fraction * len(refs))))
    hot = _clean_components(hot, max_area=p.hot_max_area)
    hot = _isolated_components(hot, p.hot_isolation)
    hot = cv2.dilate(hot.astype(np.uint8), _disk(p.hot_dilate)) > 0
    log.info("pixels quentes encontrados: %d", cv2.connectedComponents(hot.astype(np.uint8))[0] - 1)

    if calibration_dir is not None:
        return _build_from_calibration(calibration_dir, refs, shape, hot, p), best_img

    global_mask = _defect_map(refs, shape, hot, p)
    groups: dict[float, DefectMask] = {}
    for idx in _zoom_groups([r.focal for r in refs], p.zoom_group_ratio):
        members = [refs[i] for i in idx]
        covered = np.mean(np.any(np.stack([~np.isnan(r.residual[..., 0]) for r in members]), axis=0))
        if covered < p.zoom_min_own_coverage:
            continue  # poucas referências lisas nesse zoom: o mapa global serve melhor
        focal = float(np.median([r.focal for r in members]))  # type: ignore[type-var]
        dm = _defect_map(members, shape, hot, p)
        log.info(
            "zoom %.1f mm: %d refs, %d sessões, desfoque típico %.0f, intensidade típica %.2f",
            focal,
            len(members),
            len({r.session for r in members}),
            dm.prior_sigma,
            dm.prior_k,
        )
        groups[focal] = dm

    return MaskSet(global_mask, groups, p.zoom_group_ratio), best_img


def _build_from_calibration(
    calibration_dir: Path, refs: list[_Reference], shape: tuple[int, int], hot: np.ndarray, p: MaskParams
) -> MaskSet:
    cal: list[_Reference] = []
    for path in tqdm(list_images(calibration_dir), desc="Analisando calibração", unit="img"):
        try:
            img = read_image(path)
        except ImageReadError as exc:
            log.warning("%s", exc)
            continue
        if abs(img.shape[1] / img.shape[0] - shape[1] / shape[0]) > 0.01:
            log.warning("%s ignorada: proporção diferente do sensor", path.name)
            continue
        r = residual(img, p.analysis, full_shape=shape)
        flat = float(np.mean(~np.isnan(r[..., 0])))
        if flat < 0.3:
            log.info("%s ignorada: só %.0f%% de área lisa (textura, cena ou escura demais)", path.name, 100 * flat)
            continue
        cal.append(_Reference(path.name, r, read_focal_length(path), read_capture_time(path), read_fnumber(path)))
    if not cal:
        raise ValueError(f"nenhuma foto de calibração utilizável em {calibration_dir}")

    groups: dict[float, DefectMask] = {}
    sizes: dict[float, int] = {}
    for idx in _zoom_groups([r.focal for r in cal], p.zoom_group_ratio):
        members = [cal[i] for i in idx]
        focals = [r.focal for r in members if r.focal is not None]  # _zoom_groups só agrupa fotos com focal
        lo, hi, focal = min(focals), max(focals), float(np.median(focals))
        # Fotos reais nessa faixa de zoom calibram o desfoque/intensidade típicos.
        prior_refs = [
            r for r in refs if r.focal is not None and lo / p.zoom_group_ratio <= r.focal <= hi * p.zoom_group_ratio
        ]
        dm = _calibration_map(members, prior_refs, shape, hot, p)
        log.info(
            "zoom %.1f mm: %d fotos de calibração, desfoque típico %.0f, intensidade típica %.2f",
            focal,
            len(members),
            dm.prior_sigma,
            dm.prior_k,
        )
        groups[focal], sizes[focal] = dm, len(members)
    # Sem zoom no EXIF: usa o grupo mais bem calibrado.
    global_mask = groups[max(sizes, key=lambda f: sizes[f])]
    return MaskSet(global_mask, groups, p.zoom_group_ratio, calibrated=True)
