"""Parte el dataset de la primera ACT en entrenamiento y evaluación, como archivos separados.

Por qué en archivos y no con `episodes=[...]`: en LeRobot 0.4.1 ese filtro no
saca filas de un dataset local (ver `toolkit/README.md`), así que se entrenaría
también con los episodios de evaluación. Partido en dos datasets, además, cada
parte tiene sus propias estadísticas: la normalización sale sólo de los
episodios de entrenamiento, sin la fuga que tuvo la primera ACT.

    python -m toolkit.dataset.split_act_v1
"""

from __future__ import annotations

import json

from lerobot.datasets.dataset_tools import split_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from toolkit.paths import DATASETS_DIR, REPO_ROOT

SOURCE_REPO_ID = "local/pick_mameluco_act_v1"


def split_act_v1():
    split = json.loads((REPO_ROOT / "notes" / "act-split.json").read_text())
    source = LeRobotDataset(SOURCE_REPO_ID, root=DATASETS_DIR / SOURCE_REPO_ID)
    parts = split_dataset(
        source,
        splits={"train": split["train_episode_indices"], "heldout": split["heldout_episode_indices"]},
        output_dir=DATASETS_DIR / "local",
    )
    # split_dataset guarda cada parte en <output_dir>/<nombre>; la llevamos a la carpeta de su repo_id.
    for name, part in parts.items():
        target = DATASETS_DIR / part.repo_id
        (DATASETS_DIR / "local" / name).rename(target)
        parts[name] = LeRobotDataset(part.repo_id, root=target)
    return parts


if __name__ == "__main__":
    for name, part in split_act_v1().items():
        print(f"{name}: {part.repo_id} · {part.meta.total_episodes} episodios · {part.meta.total_frames} cuadros · {part.root}")
