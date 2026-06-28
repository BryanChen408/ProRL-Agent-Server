from __future__ import annotations

import torch


class Model(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def get_inputs() -> list[torch.Tensor]:
    return [torch.randn(16, 16)]
