from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from phonon_kit import model_manager


def test_builtin_prefers_local_frozen_and_falls_back_to_packaged(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PHONON_KIT_MODEL_CACHE", str(tmp_path))
    checkpoint = model_manager.builtin_model_path()
    assert checkpoint.name == "dpa4_0704.pt"
    assert checkpoint.is_file()

    frozen = tmp_path / "dpa4_0704.pt2"
    frozen.write_bytes(b"machine-local")
    assert model_manager.builtin_model_path() == frozen


def test_freeze_writes_known_cache_location(tmp_path: Path, monkeypatch):
    cache = tmp_path / "cache"
    source = tmp_path / "source.pt"
    source.write_bytes(b"checkpoint")
    monkeypatch.setenv("PHONON_KIT_MODEL_CACHE", str(cache))
    monkeypatch.setattr(model_manager, "_dp_command", lambda: Path("/fake/dp"))

    def fake_run(command, **kwargs):
        output = Path(command[command.index("-o") + 1])
        output.write_bytes(b"locally-frozen")
        return SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr(model_manager.subprocess, "run", fake_run)
    frozen = model_manager.freeze_builtin_model(source)
    assert frozen == cache / "dpa4_0704.pt2"
    assert model_manager.builtin_model_path() == frozen
    assert frozen.with_suffix(".pt2.sha256").is_file()
