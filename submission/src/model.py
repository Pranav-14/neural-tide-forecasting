"""Placeholder PyTorch model for the final submission template."""

from __future__ import annotations

import torch


class ForecastModel(torch.nn.Module):
    """Replace this placeholder with your forecasting architecture."""

    def __init__(self) -> None:
        """Create the placeholder one-parameter model."""
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input shifted by the learned scalar bias."""
        return x + self.bias
