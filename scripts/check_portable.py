"""Verify bundled assets, patient alignment, and optional model checksums."""

import argparse
import json
import os
from pathlib import Path

import torch

from trimodal_joint.cohort import load_cohort
from trimodal_joint.config import load_config
from trimodal_joint.io import sha256
from trimodal_joint.raw import source_entries


def contained_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside portable directory: {relative}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", action="store_true", help="Also hash all final models")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    checksum_path = root / "data/asset_checksums.json"
    assets = json.loads(checksum_path.read_text())
    for relative, expected in assets.items():
        path = contained_path(root, relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Missing or changed asset: {relative}")
    config = load_config(Path("configs/noise_aug_v1.yaml"))
    for field in ["source_project", "clinical_excel", "raw_root", "imagenet_weights", "output"]:
        contained_path(root, getattr(config, field))
    records, _, names, _ = load_cohort(
        Path(config.source_project), Path(config.clinical_excel), config.split_seed
    )
    reference = json.loads(Path("data/cohort_reference.json").read_text())
    observed = [{key: r[key] for key in ["patient_id", "label", "fold"]} for r in records]
    if observed != reference["patients"] or names != reference["feature_names"]:
        raise ValueError("Patient labels, fold assignments or feature list changed")
    entries = source_entries(Path(config.source_project), Path(config.raw_root), records)
    for record in records:
        for relative in record["image_paths"]:
            contained_path(root, relative)
    for entry in entries:
        for relative in [entry["path"], *entry["markers"]]:
            contained_path(root, relative)
    if args.models:
        models = json.loads(Path("migration_models.json").read_text())
        for model in models:
            path = contained_path(root, model["path"])
            if sha256(path) != model["portable_sha256"]:
                raise ValueError(f"Model checksum mismatch: {path}")
        print(f"Verified {len(models)} final models")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; install a compatible PyTorch/driver combination")
    print(f"Verified {len(assets)} assets; {len(records)} aligned patients; {len(names)} fields")
    print(f"PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
