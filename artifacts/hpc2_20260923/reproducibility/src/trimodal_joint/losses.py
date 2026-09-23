"""Matched branch supervision plus QMF ranking or TMC evidential loss."""

import torch
from torch.nn import functional as F


def weighted_mean(values: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w = weights[y]
    return (values * w).sum() / w.sum()


def evidential_loss(
    alpha: torch.Tensor, y: torch.Tensor, epoch: int, weights: torch.Tensor
) -> torch.Tensor:
    """Expected cross entropy and annealed KL, following original TMC."""
    labels = F.one_hot(y, alpha.shape[-1]).to(alpha)
    strength = alpha.sum(-1, keepdim=True)
    ce = (labels * (strength.digamma() - alpha.digamma())).sum(-1)
    adjusted = labels + (1 - labels) * alpha
    total = adjusted.sum(-1, keepdim=True)
    k = alpha.shape[-1]
    kl = (
        total.lgamma().squeeze(-1)
        - adjusted.lgamma().sum(-1)
        - torch.lgamma(alpha.new_tensor(float(k)))
        + ((adjusted - 1) * (adjusted.digamma() - total.digamma())).sum(-1)
    )
    return weighted_mean(ce + min(1.0, epoch / 10.0) * kl, y, weights)


class QualityHistory:
    """Training-patient cumulative unweighted CE, as in QMF's RGB-D history.

    Rank using previous observations, then add current detached batch losses.
    Validation/test never update this object. Device independent and serializable.
    """

    def __init__(self, patients: int, modalities: int = 3):
        self.sums = torch.zeros(patients, modalities, dtype=torch.float64)

    def update(self, indices: torch.Tensor, losses: torch.Tensor) -> None:
        self.sums.index_add_(0, indices.detach().cpu(), losses.detach().cpu().double())

    def rank(
        self, confidence: torch.Tensor, indices: torch.Tensor, mode: str = "positive_margin"
    ) -> torch.Tensor:
        if mode not in ("positive_margin", "original"):
            raise ValueError(f"Unknown ranking mode: {mode}")
        if len(indices) < 2:
            return confidence.sum() * 0
        h = self.sums[indices.detach().cpu()]
        paired = h.roll(-1, 0)
        span = (self.sums.max(0).values - self.sums.min(0).values).clamp_min(1e-12)
        margin = ((h - paired).abs() / span).to(confidence)
        sign = (paired - h).sign().to(confidence)
        if mode == "original":
            # Match upstream input shift and hinge-boundary subgradient exactly.
            target = -sign
            nonzero = torch.where(target == 0, torch.ones_like(target), target)
            loss = F.margin_ranking_loss(
                confidence,
                confidence.roll(-1, 0) + margin / nonzero,
                -target,
                margin=0.0,
                reduction="none",
            )
        else:
            loss = F.relu(margin - sign * (confidence - confidence.roll(-1, 0)))
        # Original rank reduction averages each modality over all batch pairs.
        return loss.mean(0).sum()


def total_loss(
    branches: torch.Tensor,
    out: dict,
    method: str,
    y: torch.Tensor,
    weights: torch.Tensor,
    epoch: int,
    indices: torch.Tensor,
    history: QualityHistory | None,
    rank_weight: float = 0.1,
    rank_mode: str = "positive_margin",
) -> torch.Tensor:
    if branches.shape[1] == 1:
        return F.cross_entropy(out["scores"], y, weight=weights)
    if method == "tmc":
        return evidential_loss(out["alpha"], y, epoch, weights) + sum(
            evidential_loss(out["branch_alpha"][:, m], y, epoch, weights)
            for m in range(branches.shape[1])
        )
    branch_ce = torch.stack(
        [F.cross_entropy(branches[:, m], y, reduction="none") for m in range(branches.shape[1])], 1
    )
    loss = F.cross_entropy(out["scores"], y, weight=weights) + sum(
        weighted_mean(branch_ce[:, m], y, weights) for m in range(branches.shape[1])
    )
    if method == "qmf" and rank_weight != 0:
        if history is None:
            raise ValueError("QMF requires training-only history")
        loss = loss + rank_weight * history.rank(out["confidence"], indices, mode=rank_mode)
        history.update(indices, branch_ce)
    return loss
