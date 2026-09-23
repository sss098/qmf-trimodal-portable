# Implementation checklist

- [x] Validate table labels, patient linkage, raw EMG and fixed channel audit.
- [x] Preserve original FT-Transformer and feature extraction; verify this adapter’s formulas/gradients with tests (the QMF ranking margin differs from upstream).
- [x] Implement patient sampling, fold-local preprocessing, selection/refit and resumable checkpoints.
- [x] Check all three methods with real-data smoke runs; independent code review.
- [x] Deliver locked configuration, provenance, commands and limitations.

Scope: this project only; no original source or data mutation; 100 max epochs; table labels authoritative; local ImageNet ResNet50 initialization; no task branch pretraining.

- [x] formal_v1: all 45 outer-fold runs completed.
- [x] 2026-09-16: archive exact original code, remove dead code, verify numerical parity, and clean obsolete smoke artifacts. See cleanup.md.
