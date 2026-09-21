"""Audit source linkage and build label-free raw-window cache."""

import json
import logging
from pathlib import Path

import numpy as np

from .cohort import load_cohort
from .config import Config
from .io import fingerprint, save_json, sha256
from .raw import read_segments, segment_windows, source_entries

LOG = logging.getLogger(__name__)


def code_fingerprint() -> str:
    return fingerprint({p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def prepare(config: Config) -> Path:
    source = Path(config.source_project)
    records, table, names, meta = load_cohort(
        source, Path(config.clinical_excel), config.split_seed
    )
    entries = source_entries(source, Path(config.raw_root), records)
    paths = {
        Path(config.clinical_excel),
        source / "data/blood/blood_clinical_labs.xlsx",
        source / "data/ultrasound_roi15/ultrasound_patient_manifest.csv",
        source / "data/emg/v1_1/channel_mapping_audit_forced_all87.csv",
        source / "data/emg/v1_1/patient_manifest_forced_all87.csv",
    }
    for r in records:
        paths.update(map(Path, r["image_paths"]))
    for e in entries:
        paths.add(Path(e["path"]))
        paths.update(map(Path, e["markers"]))
    sources = {str(p): sha256(p) for p in sorted(paths)}
    signature = preprocessing_signature(
        sources,
        config.split_seed,
        [Path(__file__).with_name(n) for n in ("prepare.py", "cohort.py", "raw.py")],
    )
    cache = Path(config.output) / "prepared" / signature
    manifest = cache / "manifest.json"
    if manifest.exists():
        info = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            info["fingerprint"] != signature
            or info["records"] != records
            or fingerprint(info["metadata"]) != fingerprint(meta)
        ):
            raise ValueError("Prepared manifest differs from current table labels/split/metadata")
        if info["cache_sha256"] != sha256(cache / "windows.npy") or info["table_sha256"] != sha256(
            cache / "table.npz"
        ):
            raise ValueError("Prepared cache checksum mismatch; remove corrupt cache explicitly")
        LOG.info("Validated prepared cache: %s", cache)
        return cache
    cache.mkdir(parents=True, exist_ok=True)
    temp = cache / "windows.tmp.npy"
    data = np.lib.format.open_memmap(
        temp, mode="w+", dtype=np.float32, shape=(len(records), 5, 331, 100, 6)
    )
    for i, entry in enumerate(entries):
        data[i] = segment_windows(read_segments(entry))
        LOG.info("Prepared EMG %d/%d (%s)", i + 1, len(records), entry["patient_id"])
    data.flush()
    del data
    temp.replace(cache / "windows.npy")
    np.savez(cache / "table.npz", values=table, names=np.asarray(names))
    save_json(
        manifest,
        {
            "fingerprint": signature,
            "records": records,
            "metadata": meta,
            "emg_sources": entries,
            "source_hashes": sources,
            "cache_sha256": sha256(cache / "windows.npy"),
            "table_sha256": sha256(cache / "table.npz"),
            "raw_shape": [len(records), 5, 331, 100, 6],
        },
    )
    LOG.info(
        "Prepared %d patients, %d positive; %d table fields",
        len(records),
        meta["positive"],
        len(names),
    )
    return cache


def preprocessing_signature(sources: dict, split_seed: int, implementations: list[Path]) -> str:
    """Bind prepared tensors to their source and transformation implementations."""
    return fingerprint(
        {
            "sources": sources,
            "split_seed": split_seed,
            "recipe": {p.name: sha256(p) for p in implementations},
        }
    )
