"""v1.4 fixed clinical whitelist and ID-based original table group labels."""

import re
from pathlib import Path

import numpy as np
import openpyxl
from sklearn.model_selection import StratifiedKFold

from .io import read_csv
from .raw import basename


def checked(value):
    return value is not None and str(value).strip() not in ("", "/", "无", "否", "0")


def binary_pair(row, yes, no):
    # Paired columns encode selection by presence, including literal '无' in the no column.
    def marked(v):
        return v is not None and str(v).strip() not in ("", "/")

    a, b = marked(row[yes]), marked(row[no])
    return float(a) if a != b else np.nan


def calf(value, side):
    match = re.search(side + r"(?:侧)?\s*[:：]?\s*(\d+(?:\.\d+)?)", str(value))
    return float(match[1]) if match else np.nan


def _clean_number(value: object) -> float:
    """将血液表中的范围符号和空值统一转换为浮点数/NaN。"""
    if value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(" ", ""))
    return float(match.group(0)) if match else np.nan


def excel_rows(path: Path) -> list:
    book = openpyxl.load_workbook(path, data_only=True, read_only=True)
    try:
        return list(book["Sheet1"].values)
    finally:
        book.close()


def ident(value) -> str:
    return str(int(value)) if isinstance(value, (int, float)) else str(value).strip()


def load_cohort(source: Path, clinical: Path, split_seed: int) -> tuple:
    new = excel_rows(clinical)
    old = excel_rows(source / "data/blood/blood_clinical_labs.xlsx")
    group_headers = tuple(str(v).strip() for v in new[1][1:3])
    if group_headers not in [("肌少症", "非肌少症"), ("低肌肉量组", "非低肌肉量组")]:
        raise ValueError(f"Unrecognized group columns B/C: {group_headers}")
    nr = [r for r in new[2:] if r[0] is not None]
    original = read_csv(source / "data/ultrasound_roi15/ultrasound_patient_manifest.csv")
    needed = {int(r["patient_num"]) for r in original}
    orr = [r for r in old[2:] if r[0] is not None and int(r[0]) in needed]
    nb = {ident(r[4]): r for r in nr}
    ob = {ident(r[5]): r for r in orr}
    if len(nb) != len(nr) or len(ob) != len(orr):
        raise ValueError("Duplicate table patient IDs")
    oldnum = {int(r[0]): ident(r[5]) for r in orr}
    records = []
    values = []
    for item in original:
        rec = {
            "patient_id": item["patient_id"],
            "patient_num": int(item["patient_num"]),
            "image_paths": [
                str(source / "data/ultrasound_roi15/images" / basename(v))
                for v in item["image_paths"].split("|")
                if v
            ],
        }
        if not rec["image_paths"] or any(not Path(p).is_file() for p in rec["image_paths"]):
            raise ValueError("Missing ultrasound images")
        pid = oldnum[rec["patient_num"]]
        r = nb[pid]
        blood = ob[pid]
        sex = str(r[5]).strip()
        if sex not in ("男", "女"):
            raise ValueError("Invalid sex encoding")
        group = binary_pair(r, 1, 2)
        if not np.isfinite(group):
            raise ValueError("Original Excel group is missing or conflicting")
        label = int(group)
        records.append(dict(rec, label=label))
        d = {"性别_男": float(sex == "男")}
        for col, name in {
            6: "年龄",
            11: "发病天数",
            36: "每日康复分钟",
            37: "身高",
            38: "体重",
            39: "BMI",
            42: "Morse",
            43: "FOIS",
            44: "NRS2002",
            46: "ADL",
        }.items():
            d[name] = _clean_number(r[col])
        if not 1 <= d["FOIS"] <= 7:
            d["FOIS"] = np.nan
        for c, n in [(12, "糖尿病"), (13, "高血压"), (15, "心脏病")]:
            d[n] = float(checked(r[c])) if r[c] is not None else np.nan
        for a, b, n in [
            (8, 9, "出血性卒中"),
            (24, 25, "吸烟"),
            (26, 27, "饮酒"),
            (28, 29, "卒中家族史"),
            (30, 31, "既往跌倒"),
            (32, 33, "疼痛"),
            (34, 35, "独立行走"),
            (47, 48, "胃管"),
        ]:
            d[n] = binary_pair(r, a, b)
        for c, n in [(16, "独居"), (17, "与配偶居住"), (18, "与子女居住"), (19, "集体居住")]:
            d[n] = float(checked(r[c])) if any(checked(r[k]) for k in range(16, 20)) else np.nan
        lesion = str(r[10] or "")
        for n, p in {
            "额叶": "额",
            "顶叶": "顶",
            "颞叶": "颞",
            "枕叶": "枕",
            "基底节": "基底",
            "丘脑": "丘脑",
            "脑干": "脑干|脑桥|桥脑|延髓|中脑",
            "小脑": "小脑",
            "放射冠": "放射冠",
            "内囊": "内囊",
            "脑室": "脑室",
        }.items():
            d["病灶_" + n] = float(bool(re.search(p, lesion))) if lesion else np.nan
        side = str(r[45]).strip()
        d["左侧偏瘫"] = float(side == "左侧") if side in ("左侧", "右侧") else np.nan
        d["小腿围_患侧"] = calf(r[41], side[0]) if side in ("左侧", "右侧") else np.nan
        d["小腿围_健侧"] = (
            calf(r[41], "右" if side == "左侧" else "左") if side in ("左侧", "右侧") else np.nan
        )
        for c in range(54, 111):
            name = str(old[1][c]).strip()
            if name not in ("L/H", "CK-MB/CK"):
                d["血液_" + name] = _clean_number(blood[c])
        values.append(d)

    names = list(values[0])
    raw = np.array([[d[n] for n in names] for d in values], dtype=np.float32)
    raw[~np.isfinite(raw)] = np.nan
    if len(records) != 85 or sum(r["label"] for r in records) != 62 or len(names) != 95:
        raise ValueError(
            "Cohort differs from verified 85 patients,62 positives,95 candidates; audit before training"
        )
    if len({r["patient_id"] for r in records}) != len(records):
        raise ValueError("Duplicate imaging IDs")
    labels = np.array([r["label"] for r in records])
    for fold, (_, test) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=split_seed).split(raw, labels)
    ):
        for i in test:
            records[i]["fold"] = fold
    return (
        records,
        raw,
        names,
        {
            "source_group_headers": group_headers,
            "label_names": ["非肌少症", "肌少症"],
            "label_source": "Excel group columns B/C; presence marks; no label recomputation",
            "candidate_names": names,
            "excluded": ["grip", "DXA", "group", "identity", "followup", "payment"],
            "n": len(records),
            "positive": int(labels.sum()),
            "negative": int((labels == 0).sum()),
        },
    )
