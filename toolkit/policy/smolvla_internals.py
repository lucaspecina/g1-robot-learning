"""SmolVLA desarmado en sus piezas, para mirar adentro (serie L7).

Cada función es un tramo del camino real de LeRobot 0.4.1
(`lerobot/policies/smolvla/`), en el mismo orden y con el mismo código por
debajo. Los tests verifican que, encadenadas, dan exactamente lo mismo que el
modelo entero.

Convención de LeRobot (al revés que la serie L6): `t = 1` es ruido puro y
`t = 0` es el bloque de acciones; la velocidad apunta del bloque al ruido y la
inferencia camina con `dt = -1 / pasos`.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


@dataclass
class Observation:
    """Lo que entra al modelo, ya preparado: imágenes a 512×512 en [-1, 1], texto en tokens, estado rellenado a 32."""

    images: list[Tensor]
    img_masks: list[Tensor]
    lang_tokens: Tensor
    lang_masks: Tensor
    state: Tensor


@dataclass
class Prefix:
    """La fila que procesa el VLM: [imagen | texto | estado], con sus máscaras."""

    embs: Tensor
    pad_masks: Tensor
    att_masks: Tensor
    segments: dict[str, slice]

    @property
    def att_2d(self) -> Tensor:
        return make_att_2d_masks(self.pad_masks, self.att_masks)

    @property
    def position_ids(self) -> Tensor:
        return torch.cumsum(self.pad_masks, dim=1) - 1


@dataclass
class AttentionLog:
    """Las probabilidades de atención de cada llamada, en orden (una por capa y pasada)."""

    probs: list[Tensor] = field(default_factory=list)


def prepare_observation(policy, batch: dict) -> Observation:
    """Del lote ya preprocesado (normalizado y tokenizado) a las entradas del modelo."""
    images, img_masks = policy.prepare_images(batch)
    return Observation(
        images=images,
        img_masks=img_masks,
        lang_tokens=batch[OBS_LANGUAGE_TOKENS],
        lang_masks=batch[OBS_LANGUAGE_ATTENTION_MASK],
        state=policy.prepare_state(batch),
    )


# --- La imagen -------------------------------------------------------------


def vision_patches(model, image: Tensor) -> Tensor:
    """SigLIP: la imagen de 512×512 cortada en 32×32 parches de 16 píxeles, cada uno descripto con 768 números."""
    vision = model.vlm_with_expert.get_vlm_model().vision_model
    return vision(pixel_values=image.to(dtype=vision.dtype)).last_hidden_state


def pixel_shuffle(model, patches: Tensor) -> Tensor:
    """Junta cada bloque de 4×4 parches vecinos en un solo vector: 1024 × 768 → 64 × 12288."""
    connector = model.vlm_with_expert.get_vlm_model().connector
    return connector.pixel_shuffle(patches, connector.scale_factor)


def project_image_tokens(model, shuffled: Tensor) -> Tensor:
    """Una matriz lleva cada vector de 12288 al ancho del VLM (960)."""
    return model.vlm_with_expert.get_vlm_model().connector.modality_projection(shuffled)


def image_tokens(model, image: Tensor) -> Tensor:
    """Los 64 tokens de una imagen, escalados como los escala `embed_prefix`."""
    tokens = project_image_tokens(model, pixel_shuffle(model, vision_patches(model, image)))
    # El escalado en el mismo tipo de número (bfloat16) que usa embed_prefix, para dar idéntico.
    return tokens * torch.tensor(tokens.shape[-1] ** 0.5, dtype=tokens.dtype, device=tokens.device)


# --- La fila del VLM -------------------------------------------------------


def embed_prefix(model, obs: Observation) -> Prefix:
    embs, pad_masks, att_masks = model.embed_prefix(
        obs.images, obs.img_masks, obs.lang_tokens, obs.lang_masks, state=obs.state
    )
    n_image = 64 * len(obs.images)
    n_text = obs.lang_tokens.shape[1]
    segments = {
        "imagen": slice(0, n_image),
        "texto": slice(n_image, n_image + n_text),
        "estado": slice(n_image + n_text, n_image + n_text + 1),
    }
    return Prefix(embs, pad_masks, att_masks, segments)


@contextmanager
def capture_attention(model):
    """Durante el bloque, guarda las probabilidades de atención de cada llamada (sin cambiar el resultado)."""
    vlm_with_expert = model.vlm_with_expert
    log = AttentionLog()

    def eager_attention_forward(attention_mask, batch_size, head_dim, query_states, key_states, value_states):
        # Copia de SmolVLMWithExpertModel.eager_attention_forward (LeRobot 0.4.1) que además guarda `probs`.
        num_att_heads = vlm_with_expert.num_attention_heads
        num_key_value_heads = vlm_with_expert.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads
        sequence_length = key_states.shape[1]
        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        ).reshape(batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim)
        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        ).reshape(batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim)
        query_states = query_states.to(dtype=torch.float32).transpose(1, 2)
        key_states = key_states.to(dtype=torch.float32).transpose(1, 2)
        att_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * head_dim**-0.5
        big_neg = torch.finfo(att_weights.dtype).min
        masked = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        probs = nn.functional.softmax(masked, dim=-1)
        log.probs.append(probs.detach().float().cpu())
        probs = probs.to(dtype=value_states.dtype)
        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        return att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

    vlm_with_expert.eager_attention_forward = eager_attention_forward
    try:
        yield log
    finally:
        del vlm_with_expert.eager_attention_forward


def run_prefix(model, prefix: Prefix):
    """Pasa la fila por las 16 capas del VLM y devuelve lo que queda guardado para las pasadas (K y V por capa)."""
    _, kv_cache = model.vlm_with_expert.forward(
        attention_mask=prefix.att_2d,
        position_ids=prefix.position_ids,
        past_key_values=None,
        inputs_embeds=[prefix.embs, None],
        use_cache=True,
        fill_kv_cache=True,
    )
    return kv_cache


# --- El experto y las pasadas ----------------------------------------------


def embed_suffix(model, x_t: Tensor, t: Tensor):
    """El bloque ruidoso y el momento `t` convertidos en 50 tokens del ancho del experto."""
    return model.embed_suffix(x_t, t)


def time_embedding(model, t: Tensor) -> Tensor:
    """El momento `t` como vector de senos y cosenos de muchas frecuencias (el que entra al MLP del experto)."""
    from lerobot.policies.smolvla.modeling_smolvla import create_sinusoidal_pos_embedding

    return create_sinusoidal_pos_embedding(
        t, model.vlm_with_expert.expert_hidden_size, model.config.min_period, model.config.max_period, device=t.device
    )


def velocity(model, prefix: Prefix, kv_cache, x_t: Tensor, t: Tensor) -> Tensor:
    """Una pasada del experto: la brújula en el punto `x_t` y momento `t`."""
    return model.denoise_step(prefix.pad_masks, kv_cache, x_t, t)


def sample_with_path(model, obs: Observation, noise: Tensor, num_steps: int | None = None) -> list[Tensor]:
    """Lo mismo que `VLAFlowMatching.sample_actions`, pero devuelve el bloque después de cada pasada."""
    num_steps = num_steps or model.config.num_steps
    prefix = embed_prefix(model, obs)
    kv_cache = run_prefix(model, prefix)
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=noise.device)
    time = torch.tensor(1.0, dtype=torch.float32, device=noise.device)
    x_t = noise.clone()
    path = [x_t.clone()]
    while time >= -dt / 2:
        v_t = velocity(model, prefix, kv_cache, x_t, time.expand(noise.shape[0]))
        x_t = x_t + dt * v_t
        time = time + dt
        path.append(x_t.clone())
    return path
