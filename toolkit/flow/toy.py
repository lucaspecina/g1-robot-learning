"""El juguete de la serie L6 (flow matching): esquivar un árbol.

El robot está en el origen y el árbol adelante. Cada demostración es el punto
por donde la persona pasó al costado del árbol: unos por la izquierda, otros
por la derecha, nadie por el medio. Es el caso más chico donde el promedio de
las demostraciones cae en un lugar que ninguna demostración visitó.

Todo es 2D para poder dibujarlo. Los notebooks escriben a mano lo que se quiere
aprender (pérdidas, entrenamiento, pasos de inferencia); acá sólo vive lo que
se repite en todos: los datos y el dibujo de la escena.
"""

from __future__ import annotations

import numpy as np
import torch
from matplotlib.patches import Circle

ROBOT = (0.0, 0.0)
TREE_CENTER = (0.0, 2.0)
TREE_RADIUS = 0.45
LEFT_CENTER = (-1.0, 2.0)
RIGHT_CENTER = (1.0, 2.0)
MODE_SPREAD = 0.15

LEFT_COLOR = "tab:blue"
RIGHT_COLOR = "tab:orange"


def sample_demos(n: int, left_fraction: float = 0.5, seed: int = 0) -> torch.Tensor:
    """Puntos por donde pasaron `n` demostraciones; `left_fraction` fue por la izquierda."""
    rng = np.random.default_rng(seed)
    # Proporción exacta, no sorteada: así "mitad y mitad" es un empate de verdad.
    went_left = rng.permutation(np.arange(n) < round(n * left_fraction))
    centers = np.where(went_left[:, None], LEFT_CENTER, RIGHT_CENTER)
    points = centers + rng.normal(0.0, MODE_SPREAD, size=(n, 2))
    return torch.tensor(points, dtype=torch.float32)


def side_colors(points: torch.Tensor | np.ndarray) -> list[str]:
    """Azul si el punto quedó a la izquierda del árbol, naranja si a la derecha."""
    xs = np.asarray(points)[:, 0]
    return [LEFT_COLOR if x < TREE_CENTER[0] else RIGHT_COLOR for x in xs]


def draw_scene(ax, demos: torch.Tensor | None = None, title: str | None = None, alpha: float = 0.5):
    """El robot, el árbol y, si se pasan, las demostraciones coloreadas por lado."""
    ax.add_patch(Circle(TREE_CENTER, TREE_RADIUS, color="forestgreen", alpha=0.35, zorder=0))
    ax.add_patch(Circle(TREE_CENTER, 0.12, color="saddlebrown", zorder=1))
    ax.plot(*ROBOT, marker="s", markersize=10, color="black", zorder=3)
    ax.annotate("robot", ROBOT, textcoords="offset points", xytext=(8, -4))
    ax.annotate("árbol", TREE_CENTER, textcoords="offset points", xytext=(-14, 30))
    if demos is not None:
        ax.scatter(demos[:, 0], demos[:, 1], s=10, c=side_colors(demos), alpha=alpha, zorder=2)
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-0.5, 3.0)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return ax
