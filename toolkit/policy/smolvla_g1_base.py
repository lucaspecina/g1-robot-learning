"""Copia local de `lerobot/smolvla_base` con las formas de nuestro robot.

Por qué existe: al entrenar desde un modelo preentrenado, LeRobot 0.4.1
conserva las formas de entrada y salida de su config en vez de tomarlas del
dataset (`make_policy` sólo las completa si están vacías). `smolvla_base` se
preentrenó con brazos SO-100 de 6 articulaciones y tres cámaras, así que al
generar recortaría nuestras acciones a 6 de las 26. La versión actual de
LeRobot ya toma las formas del dataset (verificado en `main` el 12-nov-2026).

Qué cambia: sólo `input_features` y `output_features` del `config.json`. Los
pesos y el resto de la config quedan iguales, y el entrenamiento es el script
oficial sin modificar. Para revertirlo: actualizar LeRobot y entrenar con
`--policy.path=lerobot/smolvla_base` directamente.

    python -m toolkit.policy.smolvla_g1_base
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from toolkit.paths import CHECKPOINTS_DIR

BASE_REPO = "lerobot/smolvla_base"
G1_BASE_DIR = CHECKPOINTS_DIR / "smolvla_base_g1"
CAMERA_KEY = "observation.images.camera1"  # la convención de SmolVLA: camera1 = vista de arriba


def make_g1_base(
    destination: Path = G1_BASE_DIR,
    state_dim: int = 26,
    action_dim: int = 26,
    image_shape: tuple[int, int, int] = (3, 360, 640),
) -> Path:
    source = Path(snapshot_download(BASE_REPO, allow_patterns=["config.json", "model.safetensors", "policy_*"]))
    destination.mkdir(parents=True, exist_ok=True)
    for file in source.iterdir():
        shutil.copy2(file.resolve(), destination / file.name)

    config_path = destination / "config.json"
    config = json.loads(config_path.read_text())
    config["input_features"] = {
        CAMERA_KEY: {"type": "VISUAL", "shape": list(image_shape)},
        "observation.state": {"type": "STATE", "shape": [state_dim]},
    }
    config["output_features"] = {"action": {"type": "ACTION", "shape": [action_dim]}}
    config_path.write_text(json.dumps(config, indent=4))
    return destination


RENAME_MAP = {"observation.images.cam_right_high": CAMERA_KEY}


def load_g1_policy(policy_dir: Path, dataset_stats: dict, device: str = "cuda"):
    """Política + procesadores con la normalización de nuestros datos.

    Arma los procesadores igual que `lerobot-train` al partir de un modelo
    preentrenado: las estadísticas del dataset reemplazan a las guardadas, y la
    cámara se renombra a `camera1`.
    """
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(policy_dir).to(device).eval()
    features = {**policy.config.input_features, **policy.config.output_features}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(policy_dir),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "normalizer_processor": {"stats": dataset_stats, "features": features, "norm_map": policy.config.normalization_mapping},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": dataset_stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        },
    )
    return policy, preprocessor, postprocessor


if __name__ == "__main__":
    print(f"Copia lista en {make_g1_base()}")
