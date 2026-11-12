"""Las piezas de `smolvla_internals`, encadenadas, dan lo mismo que el modelo entero."""

import pytest
import torch

from toolkit.policy.smolvla_g1_base import G1_BASE_DIR

pytestmark = pytest.mark.skipif(
    not (G1_BASE_DIR / "model.safetensors").exists(),
    reason="falta la copia local de smolvla_base: python -m toolkit.policy.smolvla_g1_base",
)


@pytest.fixture(scope="module")
def setup():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, resize_with_pad

    from toolkit.policy import smolvla_internals as vla

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = SmolVLAPolicy.from_pretrained(G1_BASE_DIR).to(device).eval()
    model = policy.model
    torch.manual_seed(0)
    image = resize_with_pad(torch.rand(2, 3, 360, 640, device=device), 512, 512, pad_value=0) * 2 - 1
    text = model.vlm_with_expert.processor.tokenizer(["pick up the folded coverall from the shelf\n"] * 2, return_tensors="pt")
    obs = vla.Observation(
        images=[image],
        img_masks=[torch.ones(2, dtype=torch.bool, device=device)],
        lang_tokens=text.input_ids.to(device),
        lang_masks=text.attention_mask.bool().to(device),
        state=torch.randn(2, 32, device=device),
    )
    noise = torch.randn(2, model.config.chunk_size, model.config.max_action_dim, device=device)
    return vla, model, obs, noise


@torch.no_grad()
def test_image_pieces_rebuild_the_image_part_of_the_row(setup):
    vla, model, obs, _ = setup
    prefix = vla.embed_prefix(model, obs)
    assert prefix.embs.shape[1] == 64 + obs.lang_tokens.shape[1] + 1
    torch.testing.assert_close(vla.image_tokens(model, obs.images[0]).float(), prefix.embs[:, prefix.segments["imagen"]])


@torch.no_grad()
def test_path_ends_where_sample_actions_ends(setup):
    vla, model, obs, noise = setup
    path = vla.sample_with_path(model, obs, noise)
    assert len(path) == model.config.num_steps + 1
    reference = model.sample_actions(obs.images, obs.img_masks, obs.lang_tokens, obs.lang_masks, obs.state, noise=noise.clone())
    torch.testing.assert_close(path[-1], reference)


@torch.no_grad()
def test_capturing_attention_does_not_change_the_result(setup):
    vla, model, obs, noise = setup
    plain = vla.sample_with_path(model, obs, noise)[-1]
    with vla.capture_attention(model) as log:
        captured = vla.sample_with_path(model, obs, noise)[-1]
    torch.testing.assert_close(plain, captured)
    layers = model.vlm_with_expert.num_vlm_layers
    assert len(log.probs) == layers * (1 + model.config.num_steps)  # la fila una vez + cada pasada
