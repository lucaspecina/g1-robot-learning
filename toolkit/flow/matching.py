"""La brújula de flow matching tal como se armó a mano en L6.2.

Vive acá para que los notebooks siguientes la usen sin reescribirla. La
convención es la de la serie: `t = 0` es ruido y `t = 1` es una demostración.
"""

from __future__ import annotations

import torch
from torch import nn


class VelocityNet(nn.Module):
    """La brújula: (punto, momento) → velocidad."""

    def __init__(self, dim: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=1))


def flow_matching_loss(model: VelocityNet, demos: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """Un paso de la pérdida: parejas al azar, un momento al azar, velocidad de la pareja como respuesta."""
    x1 = demos[torch.randint(len(demos), (batch_size,))]
    x0 = torch.randn_like(x1)
    t = torch.rand(batch_size, 1)
    xt = (1 - t) * x0 + t * x1
    return ((model(xt, t) - (x1 - x0)) ** 2).mean()


def train_velocity(demos: torch.Tensor, steps: int = 4000, seed: int = 0) -> VelocityNet:
    torch.manual_seed(seed)
    model = VelocityNet(dim=demos.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        loss = flow_matching_loss(model, demos)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


def ideal_velocity(x: torch.Tensor, t: float, demos: torch.Tensor) -> torch.Tensor:
    """El promedio exacto de las velocidades de todas las parejas que pasan por x en el momento t."""
    remaining = 1.0 - t
    squared_distance = ((x[:, None, :] - t * demos[None, :, :]) ** 2).sum(-1)
    weights = torch.softmax(-squared_distance / (2 * remaining**2), dim=1)
    return (weights @ demos - x) / remaining
