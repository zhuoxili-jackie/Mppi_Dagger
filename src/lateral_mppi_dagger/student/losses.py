from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ImitationLoss:
    total: torch.Tensor
    action: torch.Tensor
    first_difference: torch.Tensor
    second_difference: torch.Tensor
    valid_samples: int


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values)
    count = expanded.sum()
    if count == 0:
        return values.sum() * 0.0
    return values.masked_select(expanded).mean()


def imitation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    huber_delta: float,
    first_difference_weight: float,
    second_difference_weight: float,
) -> ImitationLoss:
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 16:
        raise ValueError(
            f"prediction and target must have identical [batch,window,16] shape, "
            f"got {prediction.shape} and {target.shape}"
        )
    if valid.shape != prediction.shape[:2]:
        raise ValueError(f"valid must have shape {prediction.shape[:2]}, got {valid.shape}")
    finite_target = torch.isfinite(target).all(dim=-1)
    valid = valid.bool() & finite_target
    safe_target = torch.nan_to_num(target)
    action_values = F.huber_loss(prediction, safe_target, delta=huber_delta, reduction="none")
    action_loss = _masked_mean(action_values, valid)

    if prediction.shape[1] >= 2:
        predicted_first = prediction[:, 1:] - prediction[:, :-1]
        target_first = safe_target[:, 1:] - safe_target[:, :-1]
        valid_first = valid[:, 1:] & valid[:, :-1]
        first_values = F.smooth_l1_loss(predicted_first, target_first, reduction="none")
        first_loss = _masked_mean(first_values, valid_first)
    else:
        first_loss = action_loss * 0.0

    if prediction.shape[1] >= 3:
        predicted_second = prediction[:, 2:] - 2.0 * prediction[:, 1:-1] + prediction[:, :-2]
        target_second = safe_target[:, 2:] - 2.0 * safe_target[:, 1:-1] + safe_target[:, :-2]
        valid_second = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
        second_values = F.smooth_l1_loss(predicted_second, target_second, reduction="none")
        second_loss = _masked_mean(second_values, valid_second)
    else:
        second_loss = action_loss * 0.0

    total = action_loss + first_difference_weight * first_loss + second_difference_weight * second_loss
    return ImitationLoss(
        total=total,
        action=action_loss,
        first_difference=first_loss,
        second_difference=second_loss,
        valid_samples=int(valid.sum().item()),
    )

