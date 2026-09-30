"""Ajustes do usuário: predefinições, arquivo ``fungusfix.toml`` e ajustes por foto.

Ordem de prioridade (o último vence): padrão do programa → predefinição (``--preset``) →
arquivo de configuração → opções da linha de comando → bloco ``[[foto]]`` que casa com o nome.
"""

from __future__ import annotations

import fnmatch
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .correct import CorrectionParams
from .model import AnalysisParams

NOISE_CAP_ON = 0.03  # valor validado para "paredes escuras"

# Cada predefinição é um conjunto de ajustes no mesmo formato do arquivo de configuração.
PRESETS: dict[str, dict[str, Any]] = {
    "padrao": {},
    "suave": {"teia": {"forca": 0.7}, "mancha": {"forca": 0.7}, "geral": {"teto_ganho": 0.25}},
    "maximo": {"teia": {"paredes_escuras": True}, "geral": {"teto_ganho": 0.6}},
}

KNOWN = {
    "geral": {"teto_ganho", "metodo"},
    "teia": {"forca", "desfoque", "paredes_escuras"},
    "mancha": {"forca", "ajuste_final", "teto"},
    "pixels_quentes": {"corrigir"},
}


class ConfigError(ValueError):
    pass


def _check(section: str, values: dict[str, Any], where: str) -> None:
    if section not in KNOWN:
        raise ConfigError(f"{where}: seção desconhecida [{section}] (válidas: {', '.join(sorted(KNOWN))})")
    extra = set(values) - KNOWN[section]
    if extra:
        raise ConfigError(f"{where}: opção desconhecida em [{section}]: {', '.join(sorted(extra))}")


def apply_settings(
    params: CorrectionParams, analysis: AnalysisParams, settings: dict[str, Any], where: str = "configuração"
) -> tuple[CorrectionParams, AnalysisParams]:
    """Aplica um dicionário no formato do arquivo (seções geral/teia/mancha/pixels_quentes)."""
    from .correct import Method

    for section, values in settings.items():
        if section in ("foto", "preset"):
            continue
        if not isinstance(values, dict):
            raise ConfigError(f"{where}: [{section}] deveria ser uma seção")
        _check(section, values, where)
        for key, v in values.items():
            if section == "geral" and key == "teto_ganho":
                params = replace(params, max_gain=float(v))
            elif section == "geral" and key == "metodo":
                params = replace(params, method=Method(v))
            elif section == "teia" and key == "forca":
                params = replace(params, thin_scale=float(v))
            elif section == "teia" and key == "desfoque":
                params = replace(params, blur=None if v == "auto" else float(v))
            elif section == "teia" and key == "paredes_escuras":
                analysis = replace(analysis, noise_cap=NOISE_CAP_ON if v else 0.0)
            elif section == "mancha" and key == "forca":
                params = replace(params, broad_scale=float(v))
            elif section == "mancha" and key == "ajuste_final":
                params = replace(params, broad_touchup=bool(v))
            elif section == "mancha" and key == "teto":
                params = replace(params, broad_k_factor=None if v is False else float(v))
            elif section == "pixels_quentes" and key == "corrigir":
                params = replace(params, hot_pixels=bool(v))
    if not 0.0 <= params.thin_scale <= 2.0 or not 0.0 <= params.broad_scale <= 2.0:
        raise ConfigError(f"{where}: 'forca' deve ficar entre 0 e 2")
    return params, analysis


@dataclass
class Settings:
    """Tudo o que o usuário pediu; resolve os parâmetros de cada foto."""

    base: list[tuple[str, dict[str, Any]]] = field(default_factory=list)  # (origem, ajustes), em ordem
    per_photo: list[tuple[str, dict[str, Any]]] = field(default_factory=list)  # (padrão do nome, ajustes)

    def resolve(
        self, filename: str, params: CorrectionParams, analysis: AnalysisParams
    ) -> tuple[CorrectionParams, AnalysisParams]:
        for where, s in self.base:
            params, analysis = apply_settings(params, analysis, s, where)
        for pattern, s in self.per_photo:
            if fnmatch.fnmatch(filename.lower(), pattern.lower()):
                params, analysis = apply_settings(params, analysis, s, f"[[foto]] {pattern}")
        return params, analysis


def load_settings(config: Path | None, preset: str | None, cli: dict[str, Any]) -> Settings:
    st = Settings()
    file_data: dict[str, Any] = {}
    if config is not None:
        try:
            file_data = tomllib.loads(config.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            hint = " (os nomes das opções não levam acento nem cedilha: forca, teto_ganho...)"
            raise ConfigError(f"{config}: {exc}{hint}") from exc
    preset = preset or file_data.get("preset")
    if preset:
        if preset not in PRESETS:
            raise ConfigError(f"predefinição desconhecida '{preset}' (válidas: {', '.join(PRESETS)})")
        st.base.append((f"predefinição {preset}", PRESETS[preset]))
    if file_data:
        st.base.append((str(config), file_data))
    if cli:
        st.base.append(("linha de comando", cli))
    for i, block in enumerate(file_data.get("foto", [])):
        pattern = block.get("arquivos")
        if not pattern:
            raise ConfigError(f"{config}: bloco [[foto]] nº {i + 1} sem 'arquivos'")
        st.per_photo.append((pattern, {k: v for k, v in block.items() if k != "arquivos"}))
    # Valida já (erros aparecem antes de processar o lote).
    st.resolve("", CorrectionParams(), AnalysisParams())
    for pattern, s in st.per_photo:
        apply_settings(CorrectionParams(), AnalysisParams(), s, f"[[foto]] {pattern}")
    return st
