from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .errors import ConfigError, ExternalProgramError


BUILTIN_CHECKPOINT_NAME = "dpa4_0704.pt"
BUILTIN_FROZEN_NAME = "dpa4_0704.pt2"


def packaged_checkpoint_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "models" / BUILTIN_CHECKPOINT_NAME


def model_cache_dir() -> Path:
    override = os.environ.get("PHONON_KIT_MODEL_CACHE")
    return Path(override).expanduser().resolve() if override else Path("~/.cache/phonon-kit/models").expanduser()


def frozen_builtin_model_path() -> Path:
    return model_cache_dir() / BUILTIN_FROZEN_NAME


def builtin_model_path() -> Path:
    frozen = frozen_builtin_model_path()
    if frozen.is_file() and frozen.stat().st_size > 0:
        return frozen
    return packaged_checkpoint_path()


def _dp_command() -> Path:
    adjacent = Path(sys.executable).resolve().parent / "dp"
    if adjacent.is_file() and os.access(adjacent, os.X_OK):
        return adjacent
    discovered = shutil.which("dp")
    if discovered:
        return Path(discovered)
    raise ExternalProgramError(
        "找不到 DeepMD `dp` 命令；请在安装了 DeepMD-kit 的环境中运行 ph model freeze"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_builtin_model(
    source: Path | None = None,
    *,
    force: bool = False,
    head: str | None = None,
) -> Path:
    source = packaged_checkpoint_path() if source is None else source.expanduser().resolve()
    if not source.is_file():
        raise ConfigError(f"待冻结的 DPA4 checkpoint 不存在: {source}")
    if source.suffix.lower() != ".pt":
        raise ConfigError(f"ph model freeze 需要 .pt checkpoint: {source}")
    destination = frozen_builtin_model_path()
    if destination.exists() and not force:
        raise ConfigError(f"本机冻结模型已存在: {destination}；如需重建请加 --force")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".freeze-", dir=destination.parent) as temporary:
        output = Path(temporary) / BUILTIN_FROZEN_NAME
        command = [
            str(_dp_command()),
            "--pt",
            "freeze",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
        if head:
            command.extend(["--head", head])
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode or not output.is_file():
            tail = "\n".join(result.stdout.splitlines()[-20:])
            raise ExternalProgramError(
                f"DeepMD 冻结失败（退出码 {result.returncode}）:\n{tail}"
            )
        os.replace(output, destination)
    digest = _sha256(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="utf-8"
    )
    return destination
