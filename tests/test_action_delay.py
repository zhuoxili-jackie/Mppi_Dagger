from __future__ import annotations

import torch

from lateral_mppi_dagger.env.action_delay import advance_action_delay


def test_zero_delay_applies_command_immediately() -> None:
    command = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    queue = torch.zeros((0, 2, 4), dtype=torch.float32)
    applied, updated = advance_action_delay(command, queue)
    torch.testing.assert_close(applied, command)
    assert updated.shape == (0, 2, 4)


def test_one_step_delay_is_fifo_and_batch_preserving() -> None:
    queued = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    command = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    queue = queued.unsqueeze(0)
    applied, updated = advance_action_delay(command, queue)
    torch.testing.assert_close(applied, queued)
    torch.testing.assert_close(updated[0], command)

    next_command = torch.tensor([[9.0, 10.0], [11.0, 12.0]])
    next_applied, next_queue = advance_action_delay(next_command, updated)
    torch.testing.assert_close(next_applied, command)
    torch.testing.assert_close(next_queue[0], next_command)
