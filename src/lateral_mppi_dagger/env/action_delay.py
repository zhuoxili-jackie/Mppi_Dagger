from __future__ import annotations

import torch


def advance_action_delay(
    commanded_action: torch.Tensor,
    queue: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the action applied now and the updated FIFO delay queue.

    ``queue`` has shape ``[delay_steps, batch, action_dim]``.  A zero-length
    queue is the no-delay case.  The returned tensors never alias the input
    queue, which makes the helper safe for MPPI snapshot rollouts.
    """
    if commanded_action.ndim != 2:
        raise ValueError("commanded_action must have shape [batch, action_dim].")
    if queue.ndim != 3:
        raise ValueError("queue must have shape [delay_steps, batch, action_dim].")
    if tuple(queue.shape[1:]) != tuple(commanded_action.shape):
        raise ValueError(
            "Action-delay queue batch/action shape does not match command: "
            f"{tuple(queue.shape[1:])} != {tuple(commanded_action.shape)}"
        )
    if queue.shape[0] == 0:
        return commanded_action.clone(), queue.clone()
    applied = queue[0].clone()
    updated = torch.cat((queue[1:], commanded_action.unsqueeze(0)), dim=0)
    return applied, updated
