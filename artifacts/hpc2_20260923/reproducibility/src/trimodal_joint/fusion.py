"""QMF RGB-D dynamic logits and TMC Dempster--Shafer evidence fusion."""

import torch
from torch.nn import functional as F

METHODS = ("equal_late", "qmf", "tmc")


def combine_evidence(alpha_a: torch.Tensor, alpha_b: torch.Tensor) -> torch.Tensor:
    """Combine two Dirichlet opinions using the algebraically identical DS rule.

    Expanding b/u in the original DS equations cancels (1-conflict), giving
    e_f = e_a + e_b + e_a*e_b/K. This avoids division by nearly zero conflict
    denominators while preserving the original gradients and vacuous identity.
    """
    e_a, e_b = alpha_a - 1, alpha_b - 1
    return 1 + e_a + e_b + e_a * e_b / alpha_a.shape[-1]


def fuse(scores: torch.Tensor, method: str) -> dict[str, torch.Tensor]:
    """Consume [patients, active modalities, two classes]."""
    if scores.ndim != 3 or scores.shape[1] not in (1, 2, 3) or scores.shape[2] != 2:
        raise ValueError(f"Expected [B,M,2], M in 1..3, got {tuple(scores.shape)}")
    if scores.shape[1] == 1 and method != "equal_late":
        raise ValueError("Single-modality classification requires equal_late (identity)")
    if method == "equal_late":
        fused = scores.mean(1)
        return {"scores": fused, "probabilities": fused.softmax(-1)}
    if method == "qmf":
        confidence = torch.logsumexp(scores, -1) / 10
        fused = (scores * confidence.unsqueeze(-1)).sum(1)
        return {"scores": fused, "probabilities": fused.softmax(-1), "confidence": confidence}
    if method == "tmc":
        alpha = F.softplus(scores) + 1
        fused = alpha[:, 0]
        for m in range(1, scores.shape[1]):
            fused = combine_evidence(fused, alpha[:, m])
        return {
            "alpha": fused,
            "branch_alpha": alpha,
            "probabilities": fused / fused.sum(-1, keepdim=True),
            "uncertainty": 2 / fused.sum(-1),
        }
    raise ValueError(f"Unknown fusion method: {method}")
