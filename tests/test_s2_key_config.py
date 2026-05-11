from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "configure_s2_api_key.py"
    spec = importlib.util.spec_from_file_location("configure_s2_api_key", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_s2_key_strips_env_assignment_and_smart_quotes() -> None:
    config = _load_module()

    key = config.parse_s2_api_key("  S2_API_KEY=\u201csk-test_123\u201d\n")

    assert key == "sk-test_123"


def test_zshrc_marker_block_update_is_idempotent(tmp_path: Path) -> None:
    config = _load_module()
    zshrc = tmp_path / ".zshrc"
    zshrc.write_text("export PATH=/usr/local/bin:$PATH\n", encoding="utf-8")

    config.update_zshrc(zshrc, "first-key")
    config.update_zshrc(zshrc, "second-key")

    text = zshrc.read_text(encoding="utf-8")
    assert text.count(config.START_MARKER) == 1
    assert text.count(config.END_MARKER) == 1
    assert "first-key" not in text
    assert "second-key" in text
    assert text.startswith("export PATH=/usr/local/bin:$PATH\n")


def test_default_zshrc_permission_error_falls_back_to_zshenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _load_module()
    zshrc = tmp_path / ".zshrc"
    zshenv = tmp_path / ".zshenv"
    calls: list[Path] = []
    real_update = config.update_zshrc

    def fake_update(path: Path, key: str) -> None:
        calls.append(path)
        if path == zshrc:
            raise PermissionError("read-only zshrc")
        real_update(path, key)

    monkeypatch.setattr(config, "DEFAULT_ZSHRC", zshrc)
    monkeypatch.setattr(config, "update_zshrc", fake_update)

    configured_path = config.update_shell_config(zshrc, "fallback-key")

    assert configured_path == zshenv
    assert calls == [zshrc, zshenv]
    assert "fallback-key" in zshenv.read_text(encoding="utf-8")


def test_cli_does_not_leak_key_to_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _load_module()
    secret = "secret-s2-key-123"
    input_file = tmp_path / "S2.txt"
    zshrc = tmp_path / ".zshrc"
    input_file.write_text(f"S2_API_KEY='{secret}'\n", encoding="utf-8")

    exit_code = config.main(["--input", str(input_file), "--zshrc", str(zshrc)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert secret not in output
    assert "S2_API_KEY" in output
    assert secret in zshrc.read_text(encoding="utf-8")


def test_invalid_non_ascii_key_is_rejected_without_leaking_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _load_module()
    invalid = "abc-\u03c0"
    input_file = tmp_path / "S2.txt"
    zshrc = tmp_path / ".zshrc"
    input_file.write_text(invalid, encoding="utf-8")

    exit_code = config.main(["--input", str(input_file), "--zshrc", str(zshrc)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert invalid not in captured.out
    assert invalid not in captured.err
    assert "ASCII" in captured.err
    assert not zshrc.exists()
