"""Linha de comando: ``build-mask`` (uma vez) e ``process`` (em lote)."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .correct import CorrectionParams, InpaintAlgorithm, Method, correct_image
from .imageio import ImageReadError, read_focal_length, read_image, write_image
from .mask import MaskParams, MaskSet, build_mask

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
    jpeg_quality: int = 95,
    overwrite: bool = False,
) -> BatchReport:
    report = BatchReport()
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
                out, k = correct_image(img, dm, params)
                write_image(dst, out, src, jpeg_quality=jpeg_quality)
                log.debug("%s: zoom=%s intensidade=%s", src.name, group, k)
                report.done.append(src.name)
            except Exception as exc:  # noqa: BLE001 - um arquivo ruim não interrompe o lote
                report.failed.append((src.name, repr(exc)))
                log.error("falha em %s: %s", src.name, exc)
    return report


def _cmd_build_mask(args: argparse.Namespace) -> int:
    params = MaskParams(
        fungus_threshold=args.fungus_threshold,
        fungus_threshold_low=args.fungus_threshold_low,
        fungus_dilate=args.fungus_dilate,
        hot_threshold=args.hot_threshold,
    )
    with logging_redirect_tqdm():
        masks, preview_base = build_mask(args.reference, params)
    masks.save(args.mask_dir, preview_base)
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
    masks = MaskSet.load(args.mask_dir)

    auto = args.strength == "auto"
    params = CorrectionParams(
        method=Method(args.method),
        algorithm=InpaintAlgorithm(args.algorithm),
        inpaint_radius=args.radius,
        auto_strength=auto,
        strength=1.0 if auto else float(args.strength),
    )
    log.info(
        "método=%s algoritmo=%s raio=%d intensidade=%s",
        params.method.value,
        params.algorithm.value,
        params.inpaint_radius,
        args.strength,
    )

    report = process_batch(args.input, args.output, masks, params, args.quality, args.overwrite)

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
            "--fungus-dilate", type=int, default=d.fungus_dilate, help=f"margem em px (padrão {d.fungus_dilate})"
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
    if args.command == "process" and args.strength != "auto":
        try:
            float(args.strength)
        except ValueError:
            log.error("--strength deve ser 'auto' ou um número")
            return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
