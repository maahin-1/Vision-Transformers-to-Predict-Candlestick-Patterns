import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

import explain

N = explain.GRID_ROWS * explain.GRID_COLS + 1


def uniform():
    return np.full((N, N), 1 / N)


def test_rollout_heat_follows_the_class_token_attention():
    last = np.zeros((N, N))
    last[:, 0] = 0.0
    last[0, 1 + 40] = 1.0  # class token reads patch 40 in the last layer
    last[1:, 1:] = np.eye(N)[1:, 1:]
    heat = explain.rollout_heat([uniform(), uniform(), last])
    assert heat.shape == (explain.GRID_ROWS, explain.GRID_COLS)
    assert heat.max() == pytest.approx(1.0) and heat.min() == pytest.approx(0.0)
    assert np.unravel_index(heat.argmax(), heat.shape) == divmod(40, explain.GRID_COLS)


def test_rollout_heat_of_flat_attention_is_all_zero_not_nan():
    heat = explain.rollout_heat([np.eye(N)])
    assert not np.isnan(heat).any() and heat.sum() == 0


def test_rollout_rejects_a_different_patch_count():
    with pytest.raises(ValueError, match='patches'):
        explain.rollout_heat([np.eye(10)])


def test_recent_share():
    heat = np.zeros((explain.GRID_ROWS, explain.GRID_COLS))
    heat[:, -3:] = 1.0
    assert explain.recent_share(heat) == pytest.approx(1.0)
    assert explain.recent_share(np.ones_like(heat)) == pytest.approx(3 / explain.GRID_COLS)
    assert explain.recent_share(np.zeros_like(heat)) == 0.0


def test_overlay_and_upscale_shapes_and_cold_regions_keep_the_picture():
    image = np.full((104, 72, 3), 200, dtype=np.uint8)
    assert explain.upscale(image).shape == (312, 216, 3)
    cold = explain.overlay(image, np.zeros((explain.GRID_ROWS, explain.GRID_COLS)))
    assert cold.shape == (312, 216, 3) and (cold == 200).all()
    hot = explain.overlay(image, np.ones((explain.GRID_ROWS, explain.GRID_COLS)))
    assert not (hot == 200).all()


def test_layer_attentions_matches_the_model_and_cleans_up():
    torch = pytest.importorskip('torch')
    from model import ViT

    model = ViT().eval()
    tensor = torch.randn(3, 104, 72)
    with torch.inference_mode():
        before = model(tensor.unsqueeze(0))
    attentions = explain.layer_attentions(model, tensor)
    with torch.inference_mode():
        after = model(tensor.unsqueeze(0))
    assert len(attentions) == 3 and attentions[0].shape == (N, N)
    assert all(np.allclose(a.sum(-1), 1, atol=1e-4) for a in attentions)
    assert torch.allclose(before, after)
    assert all(not layer.self_attention._forward_pre_hooks for layer in model.encoder.layers)
