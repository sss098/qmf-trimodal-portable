"""Read the audited six-channel raw recordings without inferring clinical labels."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch

from .io import read_csv


def basename(value: str) -> str:
    return value.replace("\\", "/").split("/")[-1]


def segment_windows(segments: np.ndarray) -> np.ndarray:
    if segments.shape != (5, 10000, 6):
        raise ValueError(f"Expected five 10s six-channel contractions: {segments.shape}")
    return np.stack([segments[:, i : i + 100] for i in range(0, 9901, 30)], axis=1).astype(
        np.float32
    )


def source_entries(source_project: Path, raw_root: Path, records: list[dict]) -> list[dict]:
    folder = source_project / "data/emg/v1_1"
    manifest = {r["patient_id"]: r for r in read_csv(folder / "patient_manifest_forced_all87.csv")}
    mappings = {}
    for r in read_csv(folder / "channel_mapping_audit_forced_all87.csv"):
        mappings.setdefault(r["patient_id"], []).append(r)
    directories = {}
    for p in (raw_root / "肌少症/肌电图").iterdir():
        m = re.match(r"\s*(\d+)\s*号", p.name)
        if p.is_dir() and m:
            number = int(m[1])
            if number in directories:
                raise ValueError("Duplicate EMG patient directory")
            directories[number] = p
    entries = []
    for record in records:
        pid = record["patient_id"]
        m = manifest[pid]
        directory = directories[record["patient_num"]]
        signal_path = resolve_signal(directory, m["processed_from_fallback"].lower() == "true")
        channel_rows = sorted(mappings[pid], key=lambda r: int(r["logical_channel"]))
        if [int(r["logical_channel"]) for r in channel_rows] != list(range(1, 7)):
            raise ValueError(f"{pid}: incomplete or duplicate logical channel audit")
        markers = []
        for value in m["marker_csvs"].split("|"):
            if value:
                matched = list(directory.rglob(basename(value)))
                if len(matched) != 1:
                    raise ValueError(f"{pid}: missing marker")
                markers.append(str(matched[0]))
        entries.append(
            {
                "patient_id": pid,
                "path": str(signal_path),
                "markers": markers,
                "prefiltered": m["processed_from_fallback"].lower() == "true",
                "columns": [r["physical_csv_column"] for r in channel_rows],
                "quality_flags": m["quality_flags"],
            }
        )
    return entries


def read_segments(entry: dict) -> np.ndarray:
    frame = pd.read_csv(entry["path"], header=None, skiprows=1, encoding="utf-8-sig")
    columns = entry["columns"]
    observed = [int(c) - 1 for c in columns if c.isdigit()]
    if not observed or max(observed) >= frame.shape[1]:
        raise ValueError("Invalid channel columns")
    signal = frame.iloc[:, observed].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(signal).all():
        raise ValueError("Nonfinite source waveform")
    if not entry["prefiltered"]:
        b, a = butter(3, [10 / 500, 499 / 500], btype="bandpass")
        signal = filtfilt(b, a, signal, axis=0)
        b, a = iirnotch(50 / 500, 35)
        signal = filtfilt(b, a, signal, axis=0)
    ordered = []
    pos = 0
    for c in columns:
        if c.isdigit():
            ordered.append(signal[:, pos])
            pos += 1
        elif c.startswith("imputed_median"):
            ordered.append(np.median(signal, axis=1))
        else:
            raise ValueError(f"Unknown audited channel value: {c}")
    signal = np.stack(ordered, 1)
    if entry["prefiltered"]:
        if len(entry["markers"]) != 5:
            raise ValueError("Expected five fallback markers")
        time = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(float)
        if not np.isfinite(time).all():
            raise ValueError("Nonfinite fallback timestamps")
        order = np.argsort(time)
        time = time[order]
        signal = signal[order]
        time, unique = np.unique(time, return_index=True)
        signal = signal[unique]
        segments = []
        for path in entry["markers"]:
            mf = pd.read_csv(path, header=None, skiprows=1, encoding="utf-8-sig", nrows=1)
            start = float(mf.iloc[0, 0])
            grid = start + np.arange(10000) / 1000
            if grid[0] < time[0] or grid[-1] > time[-1] + 0.002:
                raise ValueError("Filtered fallback does not cover contraction")
            segments.append(np.stack([np.interp(grid, time, signal[:, c]) for c in range(6)], 1))
    else:
        if len(signal) < 140000:
            raise ValueError("Raw recording shorter than fixed protocol")
        segments = [signal[start * 1000 : (start + 10) * 1000] for start in (50, 70, 90, 110, 130)]
    result = np.asarray(segments, dtype=np.float32)
    if result.shape != (5, 10000, 6) or not np.isfinite(result).all():
        raise ValueError("Invalid reconstructed waveform segments")
    return result


def resolve_signal(directory: Path, prefiltered: bool) -> Path:
    """Require one CSV in this patient's raw or prefiltered acquisition folder.

    Recorded filenames are not used: source files may have been renamed.
    """
    subdir = "滤波后数据" if prefiltered else "原始数据"
    candidates = list(directory.rglob(f"{subdir}/*.csv"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one {subdir} signal, found {len(candidates)}")
    return candidates[0]
