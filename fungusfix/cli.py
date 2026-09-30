"""Linha de comando: ``build-mask`` (uma vez) e ``process`` (em lote)."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .config import ConfigError, Settings, load_settings
from .correct import CorrectionParams, InpaintAlgorithm, Method, correct_image
from .imageio import ImageReadError, read_fnumber, read_focal_length, read_image, write_image
from .mask import MaskParams, MaskSet, build_mask
from .model import AnalysisParams

log = logging.getLogger("fungusfix")


@dataclass
class BatchReport:
    done: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (arquivo, motivo)
    failed: list[tuple[str, str]] = field(default_factory=list)


def process_batch(
    input_dir: Path,
    output_dir: Path,
    masks: MaskSet,
    params: CorrectionParams,
    analysis: AnalysisParams,
    jpeg_quality: int = 95,
    overwrite: bool = False,
    settings: Settings | None = None,
    debug_dir: Path | None = None,
) -> BatchReport:
    report = BatchReport()
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(input_dir.iterdir()) if p.is_file()]
    output_dir.mkdir(parents=True, exist_ok=True)

    with logging_redirect_tqdm():
        for src in tqdm(files, desc="Corrigindo fotos", unit="img"):
            dst = output_dir / src.name
            if dst.exists() and not overwrite:
                report.skipped.append((src.name, "já existe na saída (use --overwrite)"))
                continue
            try:
                img = read_image(src)
            except ImageReadError as exc:
                report.skipped.append((src.name, str(exc)))
                log.warning("pulando %s: não é uma imagem válida", src.name)
                continue
            try:
                dm, group = masks.select(read_focal_length(src))
                if img.shape[:2] != dm.shape:
                    report.skipped.append((src.name, f"resolução {img.shape[1]}x{img.shape[0]} != máscara"))
                    log.warning("pulando %s: resolução diferente da máscara", src.name)
                    continue
                p, a = settings.resolve(src.name, params, analysis) if settings else (params, analysis)
                out, info = correct_image(img, dm, p, a, read_fnumber(src))
                write_image(dst, out, src, jpeg_quality=jpeg_quality)
                if debug_dir is not None:
                    _write_debug(debug_dir / f"{src.stem}_alteracao.jpg", img, out)
                if info is not None:
                    log.debug(
                        "%s: zoom=%s desfoque=%.0f intensidade=%.2f (%s)",
                        src.name,
                        group,
                        info.sigma,
                        info.k,
                        "medida" if info.measured else "típica",
                    )
                report.done.append(src.name)
            except Exception as exc:  # noqa: BLE001 - um arquivo ruim não interrompe o lote
                report.failed.append((src.name, repr(exc)))
                log.error("falha em %s: %s", src.name, exc)
    return report


def _write_debug(path: Path, before: np.ndarray, after: np.ndarray) -> None:
    """Mapa do que a correção mudou: vermelho = clareou, azul = escureceu (escala ±5%), sobre a foto em cinza."""
    w = 1152
    h = round(before.shape[0] * w / before.shape[1])
    b = cv2.resize(before, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    a = cv2.resize(after, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    change = (np.log(a + 1.0) - np.log(b + 1.0)).mean(axis=2)
    t = np.clip(change / 0.05, -1.0, 1.0)
    gray = cv2.cvtColor(b.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) * 0.5 + 64
    vis = np.dstack([gray, gray, gray])
    vis[..., 2] += np.clip(t, 0, 1) * 160  # vermelho: clareou
    vis[..., 0] += np.clip(-t, 0, 1) * 160  # azul: escureceu
    cv2.imwrite(str(path), np.clip(vis, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])


def _cmd_build_mask(args: argparse.Namespace) -> int:
    params = MaskParams(
        fungus_threshold=args.fungus_threshold,
        fungus_threshold_low=args.fungus_threshold_low,
        hot_threshold=args.hot_threshold,
    )
    with logging_redirect_tqdm():
        calibration = args.calibration if args.calibration and args.calibration.is_dir() else None
        if calibration is None:
            log.info("sem pasta de calibração: mapa do fungo estimado só pelas referências")
        masks, preview_base = build_mask(args.reference, params, calibration)
    masks.save(args.mask_dir, preview_base, params)
    frac = (masks.global_mask.fungus_mask > 0).mean()
    log.info(
        "máscara salva em %s (fungo: %.1f%% do quadro; grupos de zoom: %s)",
        args.mask_dir,
        100 * frac,
        sorted(masks.groups) or "nenhum",
    )
    log.info("confira %s antes de processar o lote", args.mask_dir / "preview.jpg")
    return 0


def _cmd_process(args: argparse.Namespace) -> int:
    if not args.input.is_dir():
        log.error("pasta de entrada não existe: %s", args.input)
        return 2
    if not (args.mask_dir / "mask.png").exists():
        if args.reference is None:
            log.error("máscara não encontrada em %s; rode 'build-mask' ou passe --reference", args.mask_dir)
            return 2
        log.info("máscara não encontrada; gerando a partir de %s", args.reference)
        _cmd_build_mask(args)
    try:
        masks, analysis = MaskSet.load(args.mask_dir)
    except (FileNotFoundError, KeyError) as exc:
        log.error("máscara inválida ou de versão antiga (%s); rode 'build-mask' de novo", exc)
        return 2

    params = CorrectionParams(
        method=Method(args.method),
        algorithm=InpaintAlgorithm(args.algorithm),
        inpaint_radius=args.radius,
        strength=None if args.strength == "auto" else float(args.strength),
    )
    cli: dict[str, dict[str, object]] = {}

    def put(section: str, key: str, value: object) -> None:
        cli.setdefault(section, {})[key] = value

    if args.blur != "auto":
        put("teia", "desfoque", float(args.blur))
    if args.max_gain is not None:
        put("geral", "teto_ganho", args.max_gain)
    if args.forca_teia is not None:
        put("teia", "forca", args.forca_teia)
    if args.forca_mancha is not None:
        put("mancha", "forca", args.forca_mancha)
    if args.teto_mancha is not None:
        put("mancha", "teto", args.teto_mancha)
    if args.paredes_escuras:
        put("teia", "paredes_escuras", True)
    if args.sem_ajuste_mancha:
        put("mancha", "ajuste_final", False)
    if args.sem_pixels_quentes:
        put("pixels_quentes", "corrigir", False)
    config = args.config
    if config is None and Path("fungusfix.toml").exists():
        config = Path("fungusfix.toml")
    try:
        settings = load_settings(config, args.preset, cli)
    except (ConfigError, OSError, ValueError) as exc:
        log.error("configuração inválida: %s", exc)
        return 2
    if config is not None:
        log.info("usando configuração de %s", config)
    log.info(
        "método=%s algoritmo=%s raio=%d intensidade=%s",
        params.method.value,
        params.algorithm.value,
        params.inpaint_radius,
        args.strength,
    )

    debug_dir = args.output / "_debug" if args.debug else None
    report = process_batch(
        args.input, args.output, masks, params, analysis, args.quality, args.overwrite, settings, debug_dir
    )

    log.info(
        "concluído: %d corrigidas, %d puladas, %d com erro", len(report.done), len(report.skipped), len(report.failed)
    )
    for name, why in report.skipped:
        log.info("  pulada  %s: %s", name, why)
    for name, why in report.failed:
        log.info("  ERRO    %s: %s", name, why)
    return 1 if report.failed else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="fungusfix",
        description="Remove fungo e pixels quentes fixos (defeito da câmera) de um lote de fotos.",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="log detalhado")
    sub = ap.add_subparsers(dest="command", required=True)

    def mask_args(p: argparse.ArgumentParser, reference_default: Path | None) -> None:
        d = MaskParams()
        p.add_argument(
            "--reference",
            type=Path,
            default=reference_default,
            help="pasta com fotos de referência (fundos lisos e claros ajudam muito)",
        )
        p.add_argument(
            "--calibration",
            type=Path,
            default=Path("calibration"),
            help="pasta com fotos de campo branco desfocado, em vários zooms (padrão: calibration)",
        )
        p.add_argument("--mask-dir", type=Path, default=Path("mask"), help="onde salvar/ler a máscara (padrão: mask)")
        p.add_argument(
            "--fungus-threshold",
            type=float,
            default=d.fungus_threshold,
            help=f"atenuação que semeia um filamento, em log (padrão {d.fungus_threshold})",
        )
        p.add_argument(
            "--fungus-threshold-low",
            type=float,
            default=d.fungus_threshold_low,
            help=f"limiar baixo da histerese (padrão {d.fungus_threshold_low})",
        )
        p.add_argument(
            "--hot-threshold",
            type=int,
            default=d.hot_threshold,
            help=f"quanto um pixel quente sobressai, 0-255 (padrão {d.hot_threshold})",
        )

    b = sub.add_parser("build-mask", help="gera a máscara estática a partir das referências (rodar uma vez)")
    mask_args(b, reference_default=Path("reference"))
    b.set_defaults(func=_cmd_build_mask)

    p = sub.add_parser("process", help="aplica a máscara em todas as fotos de uma pasta")
    mask_args(p, reference_default=None)
    p.add_argument("--input", type=Path, default=Path("input"), help="pasta de entrada (padrão: input)")
    p.add_argument("--output", type=Path, default=Path("output"), help="pasta de saída (padrão: output)")
    p.add_argument(
        "--method",
        choices=[m.value for m in Method],
        default=Method.HYBRID.value,
        help="hybrid (recomendado), flatfield ou inpaint (só cv2.inpaint)",
    )
    p.add_argument(
        "--algorithm",
        choices=[a.value for a in InpaintAlgorithm],
        default=InpaintAlgorithm.TELEA.value,
        help="cv2.INPAINT_TELEA (telea) ou cv2.INPAINT_NS (ns)",
    )
    p.add_argument("--radius", type=int, default=5, help="inpaintRadius do cv2.inpaint (padrão 5)")
    p.add_argument(
        "--strength", default="auto", help="intensidade do flat-field: 'auto' (estimada por foto) ou um número, ex. 1.2"
    )
    p.add_argument(
        "--blur", default="auto", help="desfoque da sombra: 'auto' (medido por foto) ou px de análise, ex. 4"
    )
    p.add_argument(
        "--max-gain",
        type=float,
        default=None,
        help=f"teto da correção em log (padrão {CorrectionParams.max_gain} ≈ +49%%)",
    )
    g = p.add_argument_group("ajustes por defeito (também podem ir no arquivo fungusfix.toml)")
    g.add_argument("--config", type=Path, help="arquivo de configuração (padrão: ./fungusfix.toml, se existir)")
    g.add_argument("--preset", choices=["suave", "padrao", "maximo"], help="predefinição de ajustes")
    g.add_argument("--forca-teia", type=float, help="fração da correção da teia: 0 = não corrige, 1 = completa")
    g.add_argument("--forca-mancha", type=float, help="fração da correção da mancha marrom/branca (0 a 2)")
    g.add_argument("--teto-mancha", type=float, help="limita a intensidade da mancha a N vezes a da foto (ex. 1.0)")
    g.add_argument(
        "--paredes-escuras", action="store_true", help="desconta o ruído do sensor para medir paredes sob lâmpada"
    )
    g.add_argument("--sem-ajuste-mancha", action="store_true", help="desliga o ajuste final da mancha")
    g.add_argument("--sem-pixels-quentes", action="store_true", help="não remove os pixels quentes do sensor")
    g.add_argument("--debug", action="store_true", help="salva em saída/_debug um mapa do que foi alterado")
    p.add_argument("--quality", type=int, default=95, help="qualidade JPEG de saída (padrão 95)")
    p.add_argument("--overwrite", action="store_true", help="sobrescreve arquivos já existentes na saída")
    p.set_defaults(func=_cmd_process)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    if not args.verbose:
        logging.getLogger("fungusfix.mask").setLevel(logging.WARNING)
        logging.getLogger("fungusfix").setLevel(logging.INFO)
    if args.command == "process":
        for name in ("strength", "blur"):
            value = getattr(args, name)
            try:
                value == "auto" or float(value)
            except ValueError:
                log.error("--%s deve ser 'auto' ou um número", name)
                return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
