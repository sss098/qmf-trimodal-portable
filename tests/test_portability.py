"""Portable defaults must not depend on the exporting machine's home directory."""

from pathlib import Path

from trimodal_joint.config import Config, load_config


def test_defaults_and_configs_use_relative_asset_paths():
    root = Path(__file__).resolve().parents[1]
    configs = [Config(), *(load_config(p) for p in (root / "configs").glob("*.yaml"))]
    for config in configs:
        for key in ["source_project", "clinical_excel", "raw_root", "imagenet_weights", "output"]:
            assert not Path(getattr(config, key)).is_absolute()


def test_launcher_uses_selected_interpreter_from_unrelated_directory(tmp_path):
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHON_BIN=sys.executable)
    result = subprocess.run(
        ["bash", str(root / "scripts/train_noise_aug_v1.sh"), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
