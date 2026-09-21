"""
FT-Transformer for sEMG age binary classification (v1.4) — per-channel feature tokens
-----------------------------------------------------------------------------------
基于 v4.1.2 的模型架构，将任务十分类改为年龄二分类（Old vs Young）。
标签：Old=0, Young=1
"""

import os
import re
import time
import copy
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    accuracy_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pywt

    _WAVELET_AVAILABLE = True
except Exception:
    pywt = None
    _WAVELET_AVAILABLE = False

# ======================== Config ========================
DATA_ROOT = os.path.join(os.path.dirname(__file__), "train")
SAMPLING_RATE = 1000
WINDOW_MS = 100
STEP_MS = 30
WINDOW_SIZE = int(WINDOW_MS * SAMPLING_RATE / 1000)
STEP_SIZE = int(STEP_MS * SAMPLING_RATE / 1000)
N_CHANNELS = 16
N_CLASSES = 2  # Old vs Young
RANDOM_SEED = 10
TRAIN_RATIO = 0.8
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results_ft_transformer_v1_4")
os.makedirs(RESULT_DIR, exist_ok=True)

# Frequency bands in Hz for FFT/STFT features
BANDS = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 500)]
STFT_WIN = 64
STFT_HOP = 32
STFT_NFFT = 128
WAVELET_NAME = "db4"
WAVELET_LEVEL = 3
VECTOR_FORMATS = ("svg",)

# Class names for plots
CLASS_NAMES = ["Old", "Young"]


# ======================== Data Index ========================
def build_file_index(data_root: str) -> List[Dict]:
    """扫描 train/ 目录，按年龄分组：old=0, young=1"""
    index = []
    pattern = re.compile(r"^(old|young)(\d+)-task(\d+)-(\d+)\.csv$")

    for folder_name in sorted(os.listdir(data_root)):
        folder_path = os.path.join(data_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # 根据文件夹名确定年龄标签
        if folder_name.startswith("old"):
            age_label = 0
        elif folder_name.startswith("young"):
            age_label = 1
        else:
            continue

        for fname in sorted(os.listdir(folder_path)):
            match = pattern.match(fname)
            if not match:
                continue

            file_path = os.path.join(folder_path, fname)
            subject_id = folder_name
            trial_id = int(match.group(4))

            index.append(
                {
                    "file_path": file_path,
                    "subject_id": subject_id,
                    "age_label": age_label,
                    "trial_id": trial_id,
                }
            )

    n_old = sum(1 for e in index if e["age_label"] == 0)
    n_young = sum(1 for e in index if e["age_label"] == 1)
    print(f"[Index] {len(index)} files found (Old={n_old}, Young={n_young})")
    return index


# ======================== Subject-aware split ========================
def _split_subject_task_files(files: List[Dict], train_ratio: float, rng: np.random.RandomState):
    files = list(files)
    rng.shuffle(files)

    n_total = len(files)
    if n_total <= 1:
        return files, []

    n_train = int(round(n_total * train_ratio))
    n_train = min(max(1, n_train), n_total - 1)
    return files[:n_train], files[n_train:]


def split_files_subject_aware(file_index: List[Dict], train_ratio: float, seed: int):
    """按 (subject_id, age_label) 分组划分，确保同一被试同一类别的文件不跨集合"""
    rng = np.random.RandomState(seed)
    subject_task_map: Dict[Tuple[str, int], List[Dict]] = {}

    for entry in file_index:
        key = (entry["subject_id"], entry["age_label"])
        subject_task_map.setdefault(key, []).append(entry)

    train_files, test_files = [], []
    singletons = 0

    for files in subject_task_map.values():
        tr, te = _split_subject_task_files(files, train_ratio, rng)
        train_files.extend(tr)
        test_files.extend(te)
        if len(files) <= 1:
            singletons += 1

    if singletons > 0:
        print(f"[Split] {singletons} subject-age buckets have only 1 file; kept in train.")

    return train_files, test_files


def load_csv(file_path: str) -> np.ndarray:
    data = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, N_CHANNELS)
    return data


def segment_windows(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = data.shape[0]
    if n_samples < WINDOW_SIZE:
        return np.empty((0, WINDOW_SIZE, N_CHANNELS), dtype=np.float32), np.empty((0,), dtype=np.int32)

    n_windows = (n_samples - WINDOW_SIZE) // STEP_SIZE + 1
    windows = np.zeros((n_windows, WINDOW_SIZE, N_CHANNELS), dtype=np.float32)
    starts = np.zeros((n_windows,), dtype=np.int32)

    for i in range(n_windows):
        start = i * STEP_SIZE
        end = start + WINDOW_SIZE
        windows[i] = data[start:end, :]
        starts[i] = start

    return windows, starts


# ======================== Handcrafted Features ========================
def _zero_crossings(x: np.ndarray, eps: float = 1e-6) -> int:
    x1 = x[:-1]
    x2 = x[1:]
    signs = (x1 >= 0) != (x2 >= 0)
    return int(np.sum(signs & (np.abs(x1 - x2) > eps)))


def _slope_sign_changes(x: np.ndarray, eps: float = 1e-6) -> int:
    dx = np.diff(x)
    dx1 = dx[:-1]
    dx2 = dx[1:]
    signs = (dx1 >= 0) != (dx2 >= 0)
    return int(np.sum(signs & (np.abs(dx1 - dx2) > eps)))


def _fft_features(x: np.ndarray, fs: float) -> np.ndarray:
    n = x.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.fft.rfft(x)
    power = (np.abs(fft_vals) ** 2).astype(np.float64)

    band_energies = []
    for f_lo, f_hi in BANDS:
        mask = (freqs >= f_lo) & (freqs < f_hi)
        band_energies.append(power[mask].sum())

    power_sum = power.sum() + 1e-12
    centroid = float((freqs * power).sum() / power_sum)
    peak_freq = float(freqs[np.argmax(power)])

    cumsum = np.cumsum(power)
    median_freq = float(freqs[np.searchsorted(cumsum, 0.5 * power_sum)])
    mean_power_freq = centroid

    return np.array(band_energies + [centroid, peak_freq, median_freq, mean_power_freq], dtype=np.float32)


def _stft_features(x: np.ndarray, fs: float) -> np.ndarray:
    n = x.shape[0]
    if n < STFT_WIN:
        return np.zeros((len(BANDS),), dtype=np.float32)

    frames = []
    for start in range(0, n - STFT_WIN + 1, STFT_HOP):
        frame = x[start:start + STFT_WIN]
        win = np.hanning(STFT_WIN)
        frames.append(frame * win)

    if not frames:
        return np.zeros((len(BANDS),), dtype=np.float32)

    mags = []
    for frame in frames:
        spec = np.fft.rfft(frame, n=STFT_NFFT)
        mags.append(np.abs(spec) ** 2)

    mags = np.stack(mags, axis=0)
    freqs = np.fft.rfftfreq(STFT_NFFT, d=1.0 / fs)

    band_energy = []
    for f_lo, f_hi in BANDS:
        mask = (freqs >= f_lo) & (freqs < f_hi)
        band_energy.append(mags[:, mask].mean())

    return np.array(band_energy, dtype=np.float32)


def _wavelet_energy(x: np.ndarray) -> np.ndarray:
    if not _WAVELET_AVAILABLE:
        return np.zeros((WAVELET_LEVEL + 1,), dtype=np.float32)

    coeffs = pywt.wavedec(x, WAVELET_NAME, level=WAVELET_LEVEL)
    energies = [np.sum(c ** 2) for c in coeffs]
    return np.array(energies, dtype=np.float32)


def extract_features(window: np.ndarray, fs: float) -> np.ndarray:
    feats = []
    for ch in range(window.shape[1]):
        x = window[:, ch].astype(np.float32)

        rms = float(np.sqrt(np.mean(x ** 2)))
        mav = float(np.mean(np.abs(x)))
        peak = float(np.max(np.abs(x)))
        zc = float(_zero_crossings(x))
        wl = float(np.sum(np.abs(np.diff(x))))
        ssc = float(_slope_sign_changes(x))

        fft_feats = _fft_features(x, fs)
        stft_feats = _stft_features(x, fs)
        wav_feats = _wavelet_energy(x)

        feats.append(
            np.concatenate(
                [
                    np.array([rms, mav, peak, zc, wl, ssc], dtype=np.float32),
                    fft_feats,
                    stft_feats,
                    wav_feats,
                ]
            )
        )

    return np.concatenate(feats, axis=0).astype(np.float32)


def extract_features_batch(windows: np.ndarray, fs: float) -> np.ndarray:
    features = [extract_features(w, fs) for w in windows]
    return np.stack(features, axis=0)


# ======================== Normalization ========================
class ChannelZScore:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, windows_list: List[np.ndarray]) -> None:
        sum_c = np.zeros((N_CHANNELS,), dtype=np.float64)
        sumsq_c = np.zeros((N_CHANNELS,), dtype=np.float64)
        total = 0

        for w in windows_list:
            if w.size == 0:
                continue
            sum_c += w.sum(axis=(0, 1))
            sumsq_c += (w ** 2).sum(axis=(0, 1))
            total += w.shape[0] * w.shape[1]

        if total == 0:
            self.mean = np.zeros((N_CHANNELS,), dtype=np.float32)
            self.std = np.ones((N_CHANNELS,), dtype=np.float32)
            return

        mean = sum_c / total
        var = sumsq_c / total - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-12))

        self.mean = mean.astype(np.float32)
        self.std = (std + 1e-8).astype(np.float32)

    def transform(self, windows: np.ndarray) -> np.ndarray:
        return (windows - self.mean[None, None, :]) / self.std[None, None, :]


# ======================== Dataset ========================
class WindowRawDataset(Dataset):
    def __init__(self, windows: np.ndarray, features: np.ndarray, labels: np.ndarray, file_ids: List[str]):
        self.windows = windows.astype(np.float32)
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.file_ids = file_ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.windows[idx])
        f = torch.from_numpy(self.features[idx])
        y = torch.tensor(self.labels[idx]).long()
        return x, f, y, self.file_ids[idx]


def build_raw_dataset(file_entries: List[Dict], normalizer: ChannelZScore, fit_normalizer: bool) -> WindowRawDataset:
    all_windows = []
    all_labels = []
    all_file_ids = []

    for entry in file_entries:
        data = load_csv(entry["file_path"])
        windows, _ = segment_windows(data)
        if windows.shape[0] == 0:
            continue

        label = entry["age_label"]  # 0=Old, 1=Young
        all_windows.append(windows)
        all_labels.append(np.full((windows.shape[0],), label, dtype=np.int64))
        all_file_ids.extend([entry["file_path"]] * windows.shape[0])

    if len(all_windows) == 0:
        return WindowRawDataset(
            np.zeros((0, WINDOW_SIZE, N_CHANNELS), dtype=np.float32),
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            [],
        )

    if fit_normalizer:
        normalizer.fit(all_windows)

    windows = np.concatenate(all_windows, axis=0)
    windows = normalizer.transform(windows)
    labels = np.concatenate(all_labels, axis=0)

    features = extract_features_batch(windows, SAMPLING_RATE)

    return WindowRawDataset(windows, features, labels, all_file_ids)


# ======================== FT-Transformer (Per-channel feature fusion) ========================
class ChannelTokenizer(nn.Module):
    """Map each channel waveform (T) into a token embedding."""

    def __init__(self, window_size: int, d_token: int):
        super().__init__()
        self.proj = nn.Linear(window_size, d_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        return self.proj(x)


class FTTransformerRaw(nn.Module):
    """Transformer that fuses per-channel handcrafted features into channel tokens.

    Fusion strategy:
      - handcrafted features have shape (B, N_channels * per_chan_feat)
      - reshape -> (B, N_channels, per_chan_feat)
      - map each per-channel feature vector to a token embedding of size d_token
      - add this embedding to the corresponding channel token (element-wise)
      - proceed with cls_token, pos_embedding and Transformer encoder
    """

    def __init__(
        self,
        n_channels: int,
        window_size: int,
        n_classes: int = N_CLASSES,
        d_token: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ffn: int = 256,
        dropout: float = 0.2,
        feature_dim: int = 0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.tokenizer = ChannelTokenizer(window_size, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_channels + 1, d_token))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # per-channel feature mapper (if feature_dim > 0 and divisible by n_channels)
        self.per_channel_mapper = None
        self.per_chan_feat = 0
        fused_dim = d_token
        if feature_dim > 0:
            if feature_dim % n_channels != 0:
                raise ValueError("feature_dim must be divisible by n_channels for per-channel fusion")
            self.per_chan_feat = feature_dim // n_channels
            # map per-channel feature vector -> d_token embedding
            self.per_channel_mapper = nn.Sequential(
                nn.LayerNorm(self.per_chan_feat),
                nn.Linear(self.per_chan_feat, d_token),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_token, d_token),
            )
            # still use cls token as d_token
            fused_dim = d_token

        # classification head (operates on cls token only)
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, n_classes),
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        tokens = self.tokenizer(x)  # (B, C, d_token)

        if self.per_channel_mapper is not None:
            B = features.size(0)
            # features shape: (B, C * per_chan_feat) -> reshape
            feat = features.view(B, self.n_channels, self.per_chan_feat)
            feat = feat.to(tokens.dtype)
            feat_tok = self.per_channel_mapper(feat)  # (B, C, d_token)
            tokens = tokens + feat_tok

        # prepend cls token
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, C+1, d_token)
        tokens = tokens + self.pos_embedding
        out = self.transformer(tokens)
        cls_out = out[:, 0, :]

        return self.head(cls_out)


# ======================== Train / Eval ========================
def train_one_epoch(model, loader, optimizer, criterion) -> float:
    model.train()
    total_loss = 0.0

    for x, feats, y, _ in loader:
        x = x.to(DEVICE)
        feats = feats.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x, feats)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / max(1, len(loader.dataset))


def evaluate(model, loader) -> float:
    model.eval()
    file_votes: Dict[str, List[int]] = {}
    file_labels: Dict[str, int] = {}

    with torch.no_grad():
        for x, feats, y, file_ids in loader:
            x = x.to(DEVICE)
            feats = feats.to(DEVICE)
            logits = model(x, feats)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_np = y.numpy()

            for pred, label, file_id in zip(preds, y_np, file_ids):
                file_votes.setdefault(file_id, []).append(int(pred))
                file_labels[file_id] = int(label)

    file_correct = 0
    for file_id, votes in file_votes.items():
        vote = int(np.bincount(votes, minlength=N_CLASSES).argmax())
        if vote == file_labels[file_id]:
            file_correct += 1

    file_acc = file_correct / max(1, len(file_votes))
    return file_acc


def collect_file_probabilities(model, loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    file_logits: Dict[str, List[np.ndarray]] = {}
    file_votes: Dict[str, List[int]] = {}
    file_labels: Dict[str, int] = {}

    with torch.no_grad():
        for x, feats, y, file_ids in loader:
            x = x.to(DEVICE)
            feats = feats.to(DEVICE)
            logits = model(x, feats)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            y_np = y.numpy()

            for prob, pred, label, file_id in zip(probs, preds, y_np, file_ids):
                file_logits.setdefault(file_id, []).append(prob)
                file_votes.setdefault(file_id, []).append(int(pred))
                file_labels[file_id] = int(label)

    y_true, y_pred, y_proba = [], [], []
    for file_id in file_logits.keys():
        mean_proba = np.mean(np.stack(file_logits[file_id], axis=0), axis=0)
        vote = int(np.bincount(file_votes[file_id], minlength=N_CLASSES).argmax())
        y_true.append(file_labels[file_id])
        y_pred.append(vote)
        y_proba.append(mean_proba)

    return np.array(y_true), np.array(y_pred), np.array(y_proba)


def save_figure(fig: plt.Figure, out_path_png: str) -> None:
    fig.savefig(out_path_png, dpi=150, bbox_inches="tight")
    base, _ = os.path.splitext(out_path_png)
    for fmt in VECTOR_FORMATS:
        fig.savefig(f"{base}.{fmt}", format=fmt, bbox_inches="tight")


def plot_confusion_matrix(cm: np.ndarray, title: str, out_path: str) -> None:
    overall_acc = cm.diagonal().sum() / max(cm.sum(), 1)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{title}\n(Overall Accuracy: {overall_acc*100:.2f}%)",
                 fontsize=13, fontweight="bold", pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count", fontsize=11)

    ticks = list(range(N_CLASSES))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_yticklabels(CLASS_NAMES, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)

    row_sums = cm.sum(axis=1).astype(np.float64)
    row_sums[row_sums == 0] = 1.0
    thresh = cm.max() / 2.0

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            count = int(cm[i, j])
            pct = count / row_sums[i] * 100.0
            text_color = "white" if count > thresh else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center",
                    color=text_color, fontsize=11, fontweight="bold")

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_normalized_confusion_matrix(cm: np.ndarray, title: str, out_path: str) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums * 100
    overall_acc = cm.diagonal().sum() / max(cm.sum(), 1)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100)
    ax.set_title(f"{title}\n(Overall Accuracy: {overall_acc*100:.2f}%)",
                 fontsize=13, fontweight="bold", pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percentage (%)", fontsize=11)

    ticks = list(range(N_CLASSES))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_yticklabels(CLASS_NAMES, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)

    thresh = 50.0
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            val = cm_norm[i, j]
            text_color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    color=text_color, fontsize=12, fontweight="bold")

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, out_path: str) -> None:
    """二分类 ROC 曲线"""
    # y_proba[:, 1] is the probability for the positive class (Young)
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2.5,
            label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Chance (AUC = 0.5)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve (File-level, Old vs Young)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_pr_curve(y_true: np.ndarray, y_proba: np.ndarray, out_path: str) -> None:
    """二分类 Precision-Recall 曲线"""
    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
    ap = average_precision_score(y_true, y_proba[:, 1])

    # baseline = positive class ratio
    baseline = y_true.sum() / len(y_true)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="darkorange", lw=2.5,
            label=f"PR (AP = {ap:.4f})")
    ax.axhline(baseline, color="gray", ls="--", lw=1.5, alpha=0.6,
               label=f"Baseline ({baseline:.3f})")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve (File-level, Old vs Young)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    per_class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    overall_acc = accuracy_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#1f77b4", "#ff7f0e"]  # blue for Old, orange for Young
    bars = ax.bar(range(N_CLASSES), per_class_acc * 100, color=colors, edgecolor="black", linewidth=0.8,
                  width=0.5)

    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc*100:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(overall_acc * 100, color="red", ls="--", lw=1.5,
               label=f"Overall: {overall_acc*100:.1f}%")
    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_ylim([0, 105])
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlabel("Age Group", fontsize=12)
    ax.set_title("Per-Class Accuracy (File-level Majority Voting)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# ======================== Main ========================
def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    file_index = build_file_index(DATA_ROOT)
    if len(file_index) == 0:
        print("[Error] No data files found.")
        return

    if not _WAVELET_AVAILABLE:
        print("[Warn] PyWavelets not available. Wavelet energy features will be zeros.")

    train_files, test_files = split_files_subject_aware(file_index, TRAIN_RATIO, RANDOM_SEED)

    normalizer = ChannelZScore()
    train_dataset = build_raw_dataset(train_files, normalizer, fit_normalizer=True)
    test_dataset = build_raw_dataset(test_files, normalizer, fit_normalizer=False)

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("[Error] Empty dataset.")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    feature_dim = train_dataset.features.shape[1]
    model = FTTransformerRaw(
        n_channels=N_CHANNELS,
        window_size=WINDOW_SIZE,
        n_classes=N_CLASSES,
        feature_dim=feature_dim,
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    print(
        f"[Split] train files={len(train_files)}, test files={len(test_files)} | "
        f"train windows={len(train_dataset)}, test windows={len(test_dataset)}"
    )
    print(f"[Features] feature_dim={feature_dim}")

    start_time = time.time()

    epoch_losses = []
    epoch_file_acc = []

    best_acc = -1.0
    best_epoch = 0
    best_state = None
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        file_acc = evaluate(model, test_loader)
        epoch_losses.append(loss)
        epoch_file_acc.append(file_acc)
        print(f"[Epoch {epoch:02d}] loss={loss:.4f} file_acc={file_acc:.4f}")

        if file_acc > best_acc + EARLY_STOPPING_MIN_DELTA:
            best_acc = file_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= EARLY_STOPPING_PATIENCE:
            print(
                f"[Early stopping] epoch {epoch:02d} "
                f"(best_epoch={best_epoch:02d}, best_acc={best_acc:.4f})"
            )
            break

    elapsed = time.time() - start_time
    print(f"[Summary] best_acc={best_acc:.4f} best_epoch={best_epoch:02d} "
          f"final_acc={epoch_file_acc[-1]:.4f} elapsed={elapsed:.1f}s")

    # 不使用 best_state 恢复，直接用最后一轮训练得到的模型进行评估

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epoch_losses)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epoch_file_acc)
    axes[1].set_title("File-level Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, os.path.join(RESULT_DIR, "ft_transformer_v1_4_curves.png"))
    plt.close(fig)

    y_true, y_pred, y_proba = collect_file_probabilities(model, test_loader)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))

    plot_confusion_matrix(cm, "Age Confusion Matrix (Count & %)",
                          os.path.join(RESULT_DIR, "confusion_matrix.png"))
    plot_normalized_confusion_matrix(cm, "Age Confusion Matrix (Normalized by Row)",
                                     os.path.join(RESULT_DIR, "confusion_matrix_norm.png"))
    plot_roc_curve(y_true, y_proba, os.path.join(RESULT_DIR, "roc_curve.png"))
    plot_pr_curve(y_true, y_proba, os.path.join(RESULT_DIR, "pr_curve.png"))
    plot_per_class_accuracy(y_true, y_pred, os.path.join(RESULT_DIR, "per_class_accuracy.png"))

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )
    print("\n[Classification Report]")
    print(report)

    with open(os.path.join(RESULT_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Overall Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
        f.write(f"Elapsed Time: {elapsed:.1f}s\n")
        f.write(f"Best Epoch: {best_epoch}\n\n")
        f.write(report)


if __name__ == "__main__":
    main()
